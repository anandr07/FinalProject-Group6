import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
from PIL import Image
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Tuple
from other_models.report_generator_2 import MedicalReportGenerator
from torchvision import transforms
import re

from rouge_score import rouge_scorer


class RougeEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def compute_rouge_l(self, predictions, references):
        """
        Compute ROUGE-L score for a batch of predictions and references

        Args:
            predictions (List[str]): List of generated report texts
            references (List[str]): List of ground truth report texts

        Returns:
            float: Average ROUGE-L F1 score
        """
        scores = []
        for pred, ref in zip(predictions, references):
            # Skip empty predictions/references
            if not pred or not ref:
                continue

            score = self.scorer.score(ref, pred)
            scores.append(score['rougeL'].fmeasure)

        return sum(scores) / len(scores) if scores else 0.0

class ChestXrayDataset(Dataset):
    def __init__(self, report_df, labels_df, image_dir):
        super().__init__()
        self.image_dir = image_dir

        # Initialize image processor and transforms
        self.image_processor = AutoFeatureExtractor.from_pretrained("aehrc/cxrmate")
        self.transforms = transforms.Compose([
            transforms.Resize(size=self.image_processor.size['shortest_edge']),
            transforms.CenterCrop(size=[
                self.image_processor.size['shortest_edge'],
                self.image_processor.size['shortest_edge'],
            ]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.image_processor.image_mean,
                std=self.image_processor.image_std,
            ),
        ])

        self.label_columns = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]

        merged_df = report_df.merge(labels_df, on='image_id', how='left')
        self.df = self._filter_valid_findings(merged_df)

    def _filter_valid_findings(self, df):
        def has_alphabets(text):
            if pd.isna(text):
                return False
            return bool(re.search('[a-zA-Z]', str(text)))

        valid_df = df[df['findings'].apply(has_alphabets)].reset_index(drop=True)

        for col in self.label_columns:
            if col in valid_df.columns:
                valid_df[col] = valid_df[col].fillna(0)
                valid_df[col] = valid_df[col].apply(lambda x: max(0, float(x)))
            else:
                valid_df[col] = 0.0

        return valid_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        findings = str(row['findings'])

        labels = torch.tensor([
            float(row[col]) if col in self.df.columns else 0.0
            for col in self.label_columns
        ], dtype=torch.float)

        # Load and process image
        image_path = os.path.join(self.image_dir, row['image_id'])
        try:
            image = Image.open(image_path).convert('RGB')
            # Process image with a batch dim
            image_tensor = self.transforms(image).unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f"Error processing image {row['image_id']}: {str(e)}")
            raise

        return {
            'image': image_tensor,
            'text': findings,
            'labels': labels
        }


def train_model(report_csv_path: str, labels_csv_path: str, image_dir: str, num_epochs: int = 10,
                batch_size: int = 8):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Initialize CXRMate model and tokenizer
        logger.info("Loading CXRMate model and tokenizer")
        model = AutoModel.from_pretrained("aehrc/cxrmate", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("aehrc/cxrmate", trust_remote_code=True)

        # Get embedding dimension and setup model
        hidden_size = 768  # Default size for vision transformers
        logger.info(f"Using embedding dimension: {hidden_size}")
        model = model.to(device)
        model.train()

        # Initialize report generator
        report_generator = MedicalReportGenerator(input_embedding_dim=hidden_size)
        report_generator = report_generator.to(device)

        # Load dataset
        report_df = pd.read_csv(report_csv_path)
        labels_df = pd.read_csv(labels_csv_path)

        train_dataset = ChestXrayDataset(report_df, labels_df, image_dir)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Setup optimizers
        model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        peft_params = [p for p in report_generator.model.parameters() if p.requires_grad]
        generator_optimizer = torch.optim.AdamW([
            {'params': peft_params, 'lr': 2e-5},
            {'params': report_generator.input_projection.parameters(), 'lr': 1e-4}
        ])

        # Training loop
        rouge_evaluator = RougeEvaluator()

        for epoch in range(num_epochs):
            model.train()
            report_generator.train()

            train_losses = []
            gen_losses = []

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Properly extract tensors from batch dictionary
                    images = batch['image'].to(device)  # Changed from list comprehension
                    texts = batch['text']  # This should already be a list
                    labels = batch['labels'].to(device)  # This should already be a tensor

                    # Get image features from CXRMate's encoder
                    outputs = model.encoder(pixel_values=images)
                    image_features = outputs.last_hidden_state.mean(dim=1)

                    # Process text for report generator
                    text_encoding = report_generator.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(device)

                    # Generate report
                    gen_loss, logits = report_generator(
                        input_embeddings=image_features,
                        labels=labels,
                        target_ids=text_encoding['input_ids']
                    )

                    total_loss = gen_loss

                    # Optimization step
                    generator_optimizer.zero_grad()
                    model_optimizer.zero_grad()
                    total_loss.backward()
                    generator_optimizer.step()
                    model_optimizer.step()

                    # Track losses
                    if not torch.isnan(total_loss):
                        train_losses.append(total_loss.item())
                        gen_losses.append(gen_loss.item())

                    progress_bar.set_postfix_str(f'Loss={total_loss.item():.4f}')

                    # Sample generation
                    if batch_idx % 50 == 0:
                        with torch.no_grad():
                            sample_report = report_generator.generate_report(
                                input_embeddings=image_features[0:1],
                                labels=labels[0:1]
                            )[0]

                            # Calculate ROUGE-L score
                            rouge_l_score = rouge_evaluator.compute_rouge_l(
                                [sample_report],
                                [texts[0]]
                            )

                            logger.info("\nSample Generation:")
                            logger.info(f"Generated: {sample_report}")
                            logger.info(f"Target: {texts[0]}")
                            logger.info(f"ROUGE-L Score: {rouge_l_score:.4f}\n")

                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    logger.error(f"Error details: {str(e)}")
                    continue

            progress_bar.close()

            # Save checkpoints and log metrics
            if train_losses:
                checkpoint_dir = Path("checkpoints_cxr_mate_biogpt_rogue")
                checkpoint_dir.mkdir(exist_ok=True)

                avg_loss = sum(train_losses) / len(train_losses)

                # In the checkpoint saving section:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'report_generator_state_dict': report_generator.state_dict(),
                    'model_optimizer_state_dict': model_optimizer.state_dict(),
                    'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                    'loss': avg_loss,
                    'rouge_l_score': rouge_l_score,  # Add this line
                }

                torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt')
                logger.info(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    report_csv_path = "../../Data/final_cleaned.csv"
    image_dir = "/home/ubuntu/new_code/CheXbert/src/image_classifier/split_data/images"
    labels_csv_path = "/home/ubuntu/new_code/CheXbert/src/datasets/labeled_reports_with_images.csv"
    train_model(report_csv_path, labels_csv_path, image_dir, num_epochs=20, batch_size=8)