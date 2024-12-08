# This is the same as biomed_gpt_2 which trains along with labels, but makes the image encoder of biomed_gpt trainable.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor, AutoModel, get_linear_schedule_with_warmup
from PIL import Image
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Tuple
from report_generator_2 import MedicalReportGenerator
import open_clip
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
import re
from rouge_score import rouge_scorer

# Add this function to calculate ROUGE scores
def calculate_rouge_scores(predictions, targets):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = []

    for pred, target in zip(predictions, targets):
        # Calculate ROUGE-L score
        scores = scorer.score(pred, target)
        rouge_l_scores.append(scores['rougeL'].fmeasure)

    return np.mean(rouge_l_scores)


def get_dataloaders(
        df: pd.DataFrame,
        image_dir: str,
        processor: AutoProcessor,
        batch_size: int = 8,
        train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    # Split data into train and validation
    train_size = int(len(df) * train_split)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # Create datasets
    train_dataset = ChestXrayDataset(train_df, image_dir, processor)
    val_dataset = ChestXrayDataset(val_df, image_dir, processor)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# class ChestXrayDataset(Dataset):
#     def __init__(self, df, image_dir, preprocess_fn):
#         self.df = df
#         self.image_dir = image_dir
#         self.preprocess = preprocess_fn
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#
#         # Ensure text is string
#         findings = str(row['findings']) if pd.notna(row['findings']) else ""
#
#         # Load and preprocess image
#         image_path = os.path.join(self.image_dir, row['image_id'])
#         try:
#             image = Image.open(image_path).convert('RGB')
#             image_tensor = self.preprocess(image)
#         except Exception as e:
#             print(f"Error processing image {row['image_id']}: {str(e)}")
#             raise
#
#         return {
#             'image': image_tensor,
#             'text': findings
#         }

class ChestXrayDataset(Dataset):
    def __init__(self, report_df, labels_df, image_dir, preprocess_fn):
        super().__init__()  # Call parent class initializer
        self.image_dir = image_dir
        self.preprocess = preprocess_fn

        # List of label columns
        self.label_columns = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]

        # Merge datasets based on image_id
        merged_df = report_df.merge(labels_df, on='image_id', how='left')

        # Filter and process the dataframe
        self.df = self._filter_valid_findings(merged_df)

        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(self.df)}")
        print(f"Number of label columns: {len(self.label_columns)}")

    def _filter_valid_findings(self, df):
        def has_alphabets(text):
            if pd.isna(text):
                return False
            return bool(re.search('[a-zA-Z]', str(text)))

        valid_df = df[df['findings'].apply(has_alphabets)].reset_index(drop=True)

        # Process labels: convert negative values to 0 and fill NaN with 0
        for col in self.label_columns:
            if col in valid_df.columns:
                valid_df[col] = valid_df[col].fillna(0)
                valid_df[col] = valid_df[col].apply(lambda x: max(0, float(x)))
            else:
                valid_df[col] = 0.0

        print(f"\nData Processing Statistics:")
        print(f"Total merged rows: {len(df)}")
        print(f"Valid rows after filtering: {len(valid_df)}")
        print(f"Rows dropped: {len(df) - len(valid_df)}")

        return valid_df

    def __len__(self):
        return len(self.df)  # Return the length of the processed dataframe

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        findings = str(row['findings'])  # We know it's valid from filtering

        # Get labels
        labels = torch.tensor([
            float(row[col]) if col in self.df.columns else 0.0
            for col in self.label_columns
        ], dtype=torch.float)

        # Load and preprocess image
        image_path = os.path.join(self.image_dir, row['image_id'])
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image)
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
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Initialize BiomedCLIP using open_clip
        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        logger.info(f"Loading model and transforms from {model_name}")

        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
        tokenizer = open_clip.get_tokenizer(model_name)

        # Enable training for BiomedCLIP
        model.train()
        for param in model.parameters():
            param.requires_grad = True

        model = model.to(device)
        logger.info("Successfully loaded model and transforms")

        # Initialize report generator
        report_generator = MedicalReportGenerator(input_embedding_dim=512)  # CLIP output dim is 512
        report_generator = report_generator.to(device)

        # Load and split dataset
        report_df = pd.read_csv(report_csv_path)
        labels_df = pd.read_csv(labels_csv_path)

        # Split data into train and validation (80-20 split)
        train_size = int(0.8 * len(report_df))
        train_report_df = report_df.iloc[:train_size]
        val_report_df = report_df.iloc[train_size:]

        # Print dataset statistics
        logger.info(f"Total dataset size: {len(report_df)}")
        logger.info(f"Training set size: {len(train_report_df)}")
        logger.info(f"Validation set size: {len(val_report_df)}")

        # Create datasets
        train_dataset = ChestXrayDataset(train_report_df, labels_df, image_dir, preprocess_train)
        val_dataset = ChestXrayDataset(val_report_df, labels_df, image_dir, preprocess_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Initialize ROUGE scorer
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # Optimizers with different learning rates
        model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        peft_params = [p for p in report_generator.model.parameters() if p.requires_grad]
        generator_optimizer = torch.optim.AdamW([
            {'params': peft_params, 'lr': 2e-5},
            {'params': report_generator.input_projection.parameters(), 'lr': 1e-4}
        ])

        # Schedulers
        generator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            generator_optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model_optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        # Best metric tracking
        best_rouge_l = 0.0
        patience = 5
        patience_counter = 0

        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            report_generator.train()

            train_losses = []
            clip_losses = []
            gen_losses = []

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training')

            for batch_idx, batch in enumerate(progress_bar):
                try:
                    images = batch['image'].to(device)
                    texts = batch['text']
                    labels = batch['labels'].to(device)

                    # Get image features from BiomedCLIP
                    image_features = model.encode_image(images)
                    text_features = model.encode_text(tokenizer(texts).to(device))

                    # Normalize features
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)

                    # Calculate similarity
                    logits = image_features @ text_features.t() * model.logit_scale.exp()

                    # Contrastive loss
                    contrastive_labels = torch.arange(len(images), device=device, dtype=torch.long)
                    clip_loss_value = (nn.CrossEntropyLoss()(logits, contrastive_labels) +
                                       nn.CrossEntropyLoss()(logits.t(), contrastive_labels)) / 2

                    # Process text for report generator
                    text_encoding = report_generator.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(device)

                    # Forward pass through report generator with labels
                    gen_loss, _ = report_generator(
                        input_embeddings=image_features,
                        target_ids=text_encoding['input_ids'],
                        labels=labels
                    )

                    # Combined loss with weighting
                    total_loss = gen_loss * 0.7 + clip_loss_value * 0.3  # Adjusted weights

                    # Backward pass with gradient clipping
                    generator_optimizer.zero_grad()
                    model_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(report_generator.parameters(), max_norm=1.0)
                    generator_optimizer.step()
                    model_optimizer.step()

                    # Track losses
                    if not torch.isnan(total_loss):
                        train_losses.append(total_loss.item())
                        clip_losses.append(clip_loss_value.item())
                        gen_losses.append(gen_loss.item())

                    # Update progress bar
                    progress_bar.set_postfix({
                        'Loss': f'{total_loss.item():.4f}',
                        'CLoss': f'{clip_loss_value.item():.4f}',
                        'GLoss': f'{gen_loss.item():.4f}'
                    })

                    # Print sample outputs occasionally
                    if batch_idx % 50 == 0:
                        with torch.no_grad():
                            sample_report = report_generator.generate_reports_batch(
                                input_embeddings=image_features[0:1],
                                labels=labels[0:1]
                            )[0]
                            logger.info("\nSample Generation:")
                            logger.info(f"Generated: {sample_report}")
                            logger.info(f"Target: {texts[0]}\n")

                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    continue

            # Calculate average training losses
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_clip_loss = sum(clip_losses) / len(clip_losses)
            avg_gen_loss = sum(gen_losses) / len(gen_losses)

            # Optimized Validation phase
            model.eval()
            report_generator.eval()

            val_losses = []
            val_predictions = []
            val_targets = []

            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation')

            with torch.no_grad():
                for batch in progress_bar:
                    images = batch['image'].to(device)
                    texts = batch['text']
                    labels = batch['labels'].to(device)

                    # Get image features
                    image_features = model.encode_image(images)

                    # Calculate validation loss
                    text_encoding = report_generator.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(device)

                    val_loss, _ = report_generator(
                        input_embeddings=image_features,
                        target_ids=text_encoding['input_ids'],
                        labels=labels
                    )

                    val_losses.append(val_loss.item())

                    # Generate reports for subset of validation batch
                    if len(val_predictions) < 100:  # Only collect up to 100 samples for ROUGE
                        batch_reports = report_generator.generate_reports_batch(
                            input_embeddings=image_features,
                            labels=labels
                        )
                        remaining_samples = 100 - len(val_predictions)
                        val_predictions.extend(batch_reports[:min(len(batch_reports), remaining_samples)])
                        val_targets.extend(texts[:min(len(texts), remaining_samples)])

                    progress_bar.set_postfix({'val_loss': val_loss.item()})

            # Calculate validation metrics
            avg_val_loss = sum(val_losses) / len(val_losses)
            rouge_scores = [
                rouge_scorer_obj.score(pred, target)['rougeL'].fmeasure
                for pred, target in zip(val_predictions, val_targets)
            ]
            avg_rouge_l = sum(rouge_scores) / len(rouge_scores)

            # Log all metrics
            logger.info(f"\nEpoch {epoch + 1} Stats:")
            logger.info(f"Training Loss: {avg_train_loss:.4f}")
            logger.info(f"Generator Loss: {avg_gen_loss:.4f}")
            logger.info(f"CLIP Loss: {avg_clip_loss:.4f}")
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            logger.info(f"ROUGE-L Score: {avg_rouge_l:.4f} (calculated on {len(val_predictions)} samples)")

            # Step schedulers
            generator_scheduler.step(avg_val_loss)
            model_scheduler.step(avg_clip_loss)

            # Save checkpoints
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)

            # Save best model based on ROUGE-L score
            if avg_rouge_l > best_rouge_l:
                best_rouge_l = avg_rouge_l
                patience_counter = 0
                best_checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'report_generator_state_dict': report_generator.state_dict(),
                    'model_optimizer_state_dict': model_optimizer.state_dict(),
                    'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'rouge_l_score': avg_rouge_l
                }
                torch.save(best_checkpoint, checkpoint_dir / 'best_model.pt')
                logger.info(f"New best model saved with ROUGE-L: {avg_rouge_l:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break

            # Save regular checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'report_generator_state_dict': report_generator.state_dict(),
                'model_optimizer_state_dict': model_optimizer.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'rouge_l_score': avg_rouge_l,
                'best_rouge_l': best_rouge_l
            }
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt')

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    report_csv_path = "../../Data/final_cleaned.csv"
    image_dir = "/home/ubuntu/new_code/CheXbert/src/image_classifier/split_data/images"
    labels_csv_path = "/home/ubuntu/new_code/CheXbert/src/datasets/labeled_reports_with_images.csv"
    train_model(report_csv_path, labels_csv_path, image_dir, num_epochs=20, batch_size=8)