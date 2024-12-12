# This does not train blip but it uses it as the text encoder and trains the bio gpt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import logging
import re
import numpy as np
from typing import Dict, List
from other_models.report_generator_bioclip import MedicalReportGenerator


class ChestXrayDataset(Dataset):
    def __init__(self, report_df, labels_df, image_dir, processor, tokenizer):
        super().__init__()
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer  # Add tokenizer as a parameter

        self.label_columns = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]

        # Merge datasets based on image_id
        merged_df = report_df.merge(labels_df, on='image_id', how='left')
        self.df = self._filter_valid_findings(merged_df)

        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(self.df)}")
        print(f"Number of label columns: {len(self.label_columns)}")

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

        print(f"\nData Processing Statistics:")
        print(f"Total merged rows: {len(df)}")
        print(f"Valid rows after filtering: {len(valid_df)}")
        print(f"Rows dropped: {len(df) - len(valid_df)}")

        return valid_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        findings = str(row['findings'])

        # Process image
        image = Image.open(os.path.join(self.image_dir, row['image_id'])).convert('RGB')
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # Process text using provided tokenizer instead of report_generator.tokenizer
        encoding = self.tokenizer(
            findings,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )

        # Create labels tensor
        labels = torch.tensor([
            float(row[col]) if col in self.df.columns else 0.0
            for col in self.label_columns
        ], dtype=torch.float)

        return {
            'pixel_values': pixel_values,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
            'text': findings
        }

def freeze_blip2_layers(model, num_trainable_layers=2):
    """
    Freeze all layers except the last num_trainable_layers in each component
    """
    # Freeze vision encoder layers
    for param in model.vision_model.parameters():
        param.requires_grad = False

    # Unfreeze last few layers of vision encoder
    for layer in model.vision_model.encoder.layers[-num_trainable_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Always keep the vision projection layer trainable
    for param in model.vision_model.post_layernorm.parameters():
        param.requires_grad = True

    # Freeze Qformer
    for param in model.qformer.parameters():
        param.requires_grad = False

    # Unfreeze last few layers of Qformer
    for layer in model.qformer.encoder.layer[-num_trainable_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Freeze language model
    for param in model.language_model.parameters():
        param.requires_grad = False

    # Unfreeze last few layers of language model (OPTForCausalLM structure)
    decoder_layers = model.language_model.model.decoder.layers
    for layer in decoder_layers[-num_trainable_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Keep projection layers trainable
    if hasattr(model, 'language_projection'):
        for param in model.language_projection.parameters():
            param.requires_grad = True

    # Log layer freezing info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")


def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params:,} || all params: {all_param:,} || '
        f'trainable%: {100 * trainable_params / all_param:.2f}%'
    )
    return trainable_params, all_param


def train_model(report_csv_path: str, labels_csv_path: str, image_dir: str, num_epochs: int = 10,
                batch_size: int = 4, validation_split: float = 0.1):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Initialize BLIP-2
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float32,
            device_map='auto',
            max_memory={0: "12GB"}
        )
        model.eval()

        # Initialize report generator
        vision_hidden_size = model.vision_model.config.hidden_size
        report_generator = MedicalReportGenerator(input_embedding_dim=vision_hidden_size)
        report_generator = report_generator.to(device)

        vocab_size = report_generator.tokenizer.vocab_size
        logger.info(f"BioGPT vocabulary size: {vocab_size}")

        # Create dataset
        train_dataset = ChestXrayDataset(
            pd.read_csv(report_csv_path),
            pd.read_csv(labels_csv_path),
            image_dir,
            processor,
            report_generator.tokenizer
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        # Debug first batch
        sample_batch = next(iter(train_loader))
        logger.info("\nSample batch info:")
        logger.info(f"input_ids shape: {sample_batch['input_ids'].shape}")
        logger.info(f"input_ids max value: {sample_batch['input_ids'].max()}")
        logger.info(f"input_ids min value: {sample_batch['input_ids'].min()}")
        logger.info(f"vocab size: {vocab_size}")

        # Initialize optimizer
        optimizer = torch.optim.AdamW(report_generator.parameters(), lr=2e-5)

        # Training loop
        for epoch in range(num_epochs):
            report_generator.train()
            epoch_losses = []
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                                desc=f'Epoch {epoch + 1}/{num_epochs}')

            for batch_idx, batch in progress_bar:
                try:
                    # Move to device
                    pixel_values = batch['pixel_values'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)

                    # Get image features
                    with torch.no_grad():
                        vision_outputs = model.vision_model(pixel_values)
                        image_features = vision_outputs.pooler_output

                    # Forward pass through report generator
                    loss, _ = report_generator(
                        input_embeddings=image_features,
                        target_ids=input_ids,
                        labels=labels
                    )

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(report_generator.parameters(), max_norm=1.0)
                    optimizer.step()

                    # Track and display loss
                    current_loss = loss.item()
                    epoch_losses.append(current_loss)
                    avg_loss = sum(epoch_losses) / len(epoch_losses)

                    # Update progress bar description
                    progress_bar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Avg Loss': f'{avg_loss:.4f}'
                    })

                    # Generate sample output periodically
                    if batch_idx % 100 == 0:
                        report_generator.eval()
                        with torch.no_grad():
                            sample_report = report_generator.generate_report(
                                input_embeddings=image_features[0:1],
                                labels=labels[0:1]
                            )[0]
                            logger.info(f"\nSample at batch {batch_idx}:")
                            logger.info(f"Generated: {sample_report}")
                            logger.info(f"Original: {batch['text'][0]}\n")
                        report_generator.train()

                except Exception as e:
                    logger.error(f"Batch {batch_idx} error: {str(e)}", exc_info=True)
                    continue

            # End of epoch logging
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"\nEpoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

            # Save checkpoint
            checkpoint_path = f'checkpoint_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': report_generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    except Exception as e:
        logger.error(f"Training error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    train_model(
        "../../Data/final_cleaned.csv",
        "/home/ubuntu/new_code/CheXbert/src/datasets/labeled_reports_with_images.csv",
        "/home/ubuntu/new_code/CheXbert/src/image_classifier/split_data/images",
        num_epochs=20,
        batch_size=4
    )