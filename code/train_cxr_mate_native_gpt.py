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

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
from PIL import Image
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import logging
from torchvision import transforms
import re

from rouge_score import rouge_scorer
import numpy as np


class ChestXrayDataset(Dataset):
    def __init__(self, report_df, labels_df, image_dir):
        super().__init__()
        self.image_dir = image_dir

        # Initialize image processor
        self.image_processor = AutoFeatureExtractor.from_pretrained("aehrc/cxrmate")

        # Exactly match notebook transforms
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

        # Merge and filter data
        merged_df = report_df.merge(labels_df, on='image_id', how='left')
        self.df = self._filter_valid_findings(merged_df)
        self.image_ids = self.df['image_id'].tolist()

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        row = self.df[self.df['image_id'] == image_id].iloc[0]

        # Process image following notebook exactly
        image_path = os.path.join(self.image_dir, image_id)
        try:
            # Open and convert to RGB
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transforms(image)

            # Ensure correct channel order
            if image_tensor.shape[0] != 3:
                image_tensor = image_tensor.permute(2, 0, 1)

            # Add batch dimension and verify shape
            image_tensor = image_tensor.unsqueeze(0)
            assert image_tensor.shape[1] == 3, f"Expected 3 channels but got {image_tensor.shape[1]}"

        except Exception as e:
            print(f"Error processing image {image_id}: {str(e)}")
            raise

        findings = str(row['findings'])

        labels = torch.tensor([
            float(row[col]) if col in row.index else 0.0
            for col in self.label_columns
        ], dtype=torch.float)

        return {
            'images': image_tensor,  # Shape: [1, 3, 384, 384]
            'text': findings,
            'labels': labels
        }

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
        return len(self.image_ids)


def collate_fn(batch):
    # Stack images while maintaining proper dimensions
    images = []
    texts = []
    labels = []

    for item in batch:
        images.append(item['images'])
        texts.append(item['text'])
        labels.append(item['labels'])

    images = torch.cat(images, dim=0)  # Shape: [batch_size, 3, 384, 384]
    labels = torch.stack(labels)

    return {
        'images': images,
        'text': texts,
        'labels': labels
    }

def compute_rouge_scores(predictions, targets):
    """Compute ROUGE scores for a batch of predictions and targets."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {
        'rouge1_f': [],
        'rouge2_f': [],
        'rougeL_f': [],
        'rouge1_p': [],
        'rouge2_p': [],
        'rougeL_p': [],
        'rouge1_r': [],
        'rouge2_r': [],
        'rougeL_r': []
    }

    for pred, target in zip(predictions, targets):
        # Calculate scores
        score = scorer.score(target, pred)

        # Store F1, precision, and recall scores
        scores['rouge1_f'].append(score['rouge1'].fmeasure)
        scores['rouge2_f'].append(score['rouge2'].fmeasure)
        scores['rougeL_f'].append(score['rougeL'].fmeasure)

        scores['rouge1_p'].append(score['rouge1'].precision)
        scores['rouge2_p'].append(score['rouge2'].precision)
        scores['rougeL_p'].append(score['rougeL'].precision)

        scores['rouge1_r'].append(score['rouge1'].recall)
        scores['rouge2_r'].append(score['rouge2'].recall)
        scores['rougeL_r'].append(score['rougeL'].recall)

    # Calculate means
    mean_scores = {k: np.mean(v) for k, v in scores.items()}
    return mean_scores


def log_rouge_scores(logger, scores, epoch, batch_idx):
    """Log ROUGE scores in a formatted way."""
    logger.info(f"\nROUGE Scores (Epoch {epoch + 1}, Batch {batch_idx}):")
    logger.info("F1 Scores:")
    logger.info(f"ROUGE-1: {scores['rouge1_f']:.4f}")
    logger.info(f"ROUGE-2: {scores['rouge2_f']:.4f}")
    logger.info(f"ROUGE-L: {scores['rougeL_f']:.4f}")

    logger.info("\nPrecision Scores:")
    logger.info(f"ROUGE-1: {scores['rouge1_p']:.4f}")
    logger.info(f"ROUGE-2: {scores['rouge2_p']:.4f}")
    logger.info(f"ROUGE-L: {scores['rougeL_p']:.4f}")

    logger.info("\nRecall Scores:")
    logger.info(f"ROUGE-1: {scores['rouge1_r']:.4f}")
    logger.info(f"ROUGE-2: {scores['rouge2_r']:.4f}")
    logger.info(f"ROUGE-L: {scores['rougeL_r']:.4f}")


def compute_loss(model, tokenizer, generated_sequences, target_texts, device):
    """Compute cross entropy loss between generated and target sequences"""
    # Tokenize target texts
    target_encodings = tokenizer(
        target_texts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    # Forward pass through model to get logits
    outputs = model(
        labels=target_encodings.input_ids,
        decoder_input_ids=generated_sequences
    )

    return outputs.loss

def train_model(report_csv_path: str, labels_csv_path: str, image_dir: str, num_epochs: int = 10,
                batch_size: int = 8):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Initialize model and tokenizer
        logger.info("Loading CXRMate model and tokenizer")
        model = AutoModel.from_pretrained("aehrc/cxrmate", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("aehrc/cxrmate", trust_remote_code=True)

        model = model.to(device)
        model.train()
        logger.info("Successfully loaded CXRMate model")

        # Load dataset
        report_df = pd.read_csv(report_csv_path)
        labels_df = pd.read_csv(labels_csv_path)

        train_dataset = ChestXrayDataset(report_df, labels_df, image_dir)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        for epoch in range(num_epochs):
            model.train()
            train_losses = []
            rouge_scores_list = []

            progress_bar = tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')

            for batch_idx, batch in enumerate(train_loader):
                try:
                    images = batch['images'].to(device)  # Shape: [batch_size, 3, 384, 384]
                    texts = batch['text']

                    # Get image features
                    image_outputs = model.encoder(pixel_values=images)
                    image_features = image_outputs.last_hidden_state

                    # Create decoder inputs
                    decoder_input_ids = model.tokenize_prompt(
                        previous_findings=[None] * len(texts),
                        previous_impression=[None] * len(texts),
                        tokenizer=tokenizer,
                        max_length=256
                    ).input_ids.to(device)

                    # Forward pass
                    outputs = model(
                        encoder_outputs=[image_features],
                        decoder_input_ids=decoder_input_ids,
                        return_dict=True
                    )

                    loss = outputs.loss
                    train_losses.append(loss.item())

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
                    progress_bar.update(1)

                    # Sample generation and ROUGE every 50 batches
                    if batch_idx % 50 == 0:
                        with torch.no_grad():
                            # Remove BOS token if present
                            sequences = generated.sequences
                            if torch.all(sequences[:, 0] == 1):
                                sequences = sequences[:, 1:]

                            # Split and decode
                            _, findings, impression = model.split_and_decode_sections(
                                sequences,
                                [tokenizer.bos_token_id, tokenizer.sep_token_id, tokenizer.eos_token_id],
                                tokenizer
                            )

                            # Compute ROUGE
                            rouge_scores = compute_rouge_scores(
                                generated_reports=[f"Findings: {f} Impression: {i}" for f, i in
                                                   zip(findings, impression)],
                                target_reports=texts
                            )
                            rouge_scores_list.append(rouge_scores)

                            # Log samples
                            logger.info("\nSample Generation:")
                            logger.info(f"Generated: {findings[0]}\nImpression: {impression[0]}")
                            logger.info(f"Target: {texts[0]}")
                            log_rouge_scores(logger, rouge_scores, epoch, batch_idx)

                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue

            progress_bar.close()

            # Log epoch stats
            avg_loss = np.mean(train_losses)
            avg_rouge = {
                metric: np.mean([scores[metric] for scores in rouge_scores_list])
                for metric in rouge_scores_list[0].keys()
            }

            logger.info(f"\nEpoch {epoch + 1}:")
            logger.info(f"Average Loss: {avg_loss:.4f}")
            logger.info(f"Average ROUGE-L F1: {avg_rouge['rougeL_f']:.4f}")

            # Save checkpoint
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt')

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    report_csv_path = "../../Data/final_cleaned.csv"
    image_dir = "/home/ubuntu/new_code/CheXbert/src/image_classifier/split_data/images"
    labels_csv_path = "/home/ubuntu/new_code/CheXbert/src/datasets/labeled_reports_with_images.csv"
    train_model(report_csv_path, labels_csv_path, image_dir, num_epochs=20, batch_size=8)