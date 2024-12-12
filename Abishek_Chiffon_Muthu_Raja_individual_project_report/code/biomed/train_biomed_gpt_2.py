# This trains along with labels

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
from other_models.report_generator_2 import MedicalReportGenerator
import open_clip
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import logging
import re

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

        model = model.to(device)
        logger.info("Successfully loaded model and transforms")

        # Initialize report generator

        report_generator = MedicalReportGenerator(input_embedding_dim=512)  # CLIP output dim is 512
        report_generator = report_generator.to(device)

        # Load dataset
        report_df = pd.read_csv(report_csv_path)
        labels_df = pd.read_csv(labels_csv_path)

        # Print some statistics about the merge
        print(f"Report dataset size: {len(report_df)}")
        print(f"Labels dataset size: {len(labels_df)}")
        print(f"Unique images in report dataset: {report_df['image_id'].nunique()}")
        print(f"Unique images in labels dataset: {labels_df['image_id'].nunique()}")

        # Create dataset with both dataframes
        train_dataset = ChestXrayDataset(report_df, labels_df, image_dir, preprocess_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimizers
        model_optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        # For report generator, use PEFT parameters and input_projection layer
        peft_params = [p for p in report_generator.model.parameters() if p.requires_grad]
        generator_optimizer = torch.optim.AdamW([
            {'params': peft_params, 'lr': 2e-5},
            {'params': report_generator.input_projection.parameters(), 'lr': 1e-4}
        ])
        # After creating optimizers
        generator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            generator_optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )


        # Training loop
        for epoch in range(num_epochs):
            model.train()
            report_generator.train()

            train_losses = []
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

            # In the training loop:
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    images = batch['image'].to(device)
                    texts = batch['text']
                    labels = batch['labels'].to(device)  # Move labels to device

                    # Get image features from CLIP
                    with torch.no_grad():
                        image_features = model.encode_image(images)  # Shape: (batch_size, 512)

                    # Process text
                    text_encoding = report_generator.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(device)

                    # Forward pass through report generator with labels
                    gen_loss, logits = report_generator(
                        input_embeddings=image_features,
                        labels=labels,  # Pass labels to the model
                        target_ids=text_encoding['input_ids']
                    )
                    # Track loss
                    if not torch.isnan(gen_loss):
                        train_losses.append(gen_loss.item())

                    # Backåward pass
                    generator_optimizer.zero_grad()
                    model_optimizer.zero_grad()
                    gen_loss.backward()
                    generator_optimizer.step()
                    model_optimizer.step()

                    # Update progress bar
                    progress_bar.set_postfix({'Gen Loss': f'{gen_loss.item():.4f}'})

                    # Print sample outputs occasionally
                    # In the training loop where we generate samples
                    if batch_idx % 50 == 0:
                        with torch.no_grad():
                            sample_report = report_generator.generate_report(
                                input_embeddings=image_features[0:1],
                                labels=labels[0:1]  # Add labels for the first sample
                            )[0]
                            if not sample_report:  # If generation is empty
                                logger.warning("Empty generation detected!")
                                # Adjust model parameters
                                report_generator.model.config.min_length = 30
                                report_generator.model.config.repetition_penalty = 1.2

                            logger.info("\nSample Generation:")
                            logger.info(f"Generated: {sample_report}")
                            logger.info(f"Target: {texts[0]}\n")

                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    logger.error(f"Text content: {texts}")
                    logger.error(
                        f"Image feature shape: {image_features.shape if 'image_features' in locals() else 'not computed'}")
                    continue
            # Calculate average training loss for the epoch
            avg_train_loss = sum(train_losses) / len(train_losses)
            logger.info(f"\nEpoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")
            # In the training loop, after calculating average loss
            generator_scheduler.step(avg_train_loss)
            # Save checkpoints
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'report_generator_state_dict': report_generator.state_dict(),
                'model_optimizer_state_dict': model_optimizer.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'train_loss': avg_train_loss
            }

            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt')

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    report_csv_path = "../Data/final_cleaned.csv"
    image_dir = "/home/ubuntu/new_code/CheXbert/src/image_classifier/split_data/images"
    labels_csv_path = "/home/ubuntu/new_code/CheXbert/src/datasets/labeled_reports_with_images.csv" # Your labels dataset path
    train_model(report_csv_path, labels_csv_path, image_dir, num_epochs=20, batch_size=8)