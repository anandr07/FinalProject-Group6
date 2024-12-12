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
from other_models.report_generator_2 import MedicalReportGenerator


class ChestXrayDataset(Dataset):
    def __init__(self, report_df, labels_df, image_dir, processor):
        super().__init__()
        self.image_dir = image_dir
        self.processor = processor

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

        # Get labels
        labels = torch.tensor([
            float(row[col]) if col in self.df.columns else 0.0
            for col in self.label_columns
        ], dtype=torch.float)

        # Load and process image
        image_path = os.path.join(self.image_dir, row['image_id'])
        try:
            image = Image.open(image_path).convert('RGB')

            # Process with BLIP-2 processor
            inputs = self.processor(
                images=image,
                text=findings,
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True
            )

            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'text': findings,
                'labels': labels
            }

        except Exception as e:
            print(f"Error processing image {row['image_id']}: {str(e)}")
            raise


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
                batch_size: int = 8, num_trainable_layers: int = 2, validation_split: float = 0.1):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        # Initialize BLIP-2 and processor
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

        # Freeze layers and only train the last few
        freeze_blip2_layers(model, num_trainable_layers)
        logger.info("BLIP-2 trainable parameters:")
        print_trainable_parameters(model)
        model = model.to(device)

        # Initialize report generator
        vision_hidden_size = model.vision_model.config.hidden_size
        report_generator = MedicalReportGenerator(input_embedding_dim=vision_hidden_size)
        report_generator = report_generator.to(device)

        # Load and split dataset
        report_df = pd.read_csv(report_csv_path)
        labels_df = pd.read_csv(labels_csv_path)

        # Shuffle and split the data
        total_size = len(report_df)
        indices = np.random.permutation(total_size)
        val_size = int(total_size * validation_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        train_report_df = report_df.iloc[train_indices]
        val_report_df = report_df.iloc[val_indices]

        # Create datasets
        train_dataset = ChestXrayDataset(train_report_df, labels_df, image_dir, processor)
        val_dataset = ChestXrayDataset(val_report_df, labels_df, image_dir, processor)

        logger.info(f"Training set size: {len(train_dataset)}")
        logger.info(f"Validation set size: {len(val_dataset)}")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        # Initialize optimizers and schedulers
        model_optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-5
        )

        peft_params = [p for p in report_generator.model.parameters() if p.requires_grad]
        generator_optimizer = torch.optim.AdamW([
            {'params': peft_params, 'lr': 2e-5},
            {'params': report_generator.input_projection.parameters(), 'lr': 1e-4}
        ])

        generator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            generator_optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model_optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        def validate_model():
            model.eval()
            report_generator.eval()
            val_losses = []
            val_blip_losses = []
            val_gen_losses = []

            with torch.no_grad():
                for val_batch in val_loader:
                    # Move inputs to device
                    pixel_values = val_batch['pixel_values'].to(device)
                    input_ids = val_batch['input_ids'].to(device)
                    attention_mask = val_batch['attention_mask'].to(device)
                    labels = val_batch['labels'].to(device)

                    # Get BLIP-2 features and loss
                    blip_outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    blip_loss = blip_outputs.loss

                    # Get vision features
                    vision_outputs = model.vision_model(pixel_values)
                    image_features = vision_outputs.pooler_output

                    # Get generator loss
                    gen_loss, _ = report_generator(
                        input_embeddings=image_features,
                        labels=labels,
                        target_ids=input_ids
                    )

                    # Calculate total loss
                    total_loss = gen_loss * 0.5 + blip_loss

                    val_losses.append(total_loss.item())
                    val_blip_losses.append(blip_loss.item())
                    val_gen_losses.append(gen_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_blip_loss = sum(val_blip_losses) / len(val_blip_losses)
            avg_val_gen_loss = sum(val_gen_losses) / len(val_gen_losses)

            return avg_val_loss, avg_val_blip_loss, avg_val_gen_loss

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            report_generator.train()

            train_losses = []
            blip_losses = []
            gen_losses = []

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Move inputs to device
                    pixel_values = batch['pixel_values'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    # Get BLIP-2 features and loss
                    blip_outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    blip_loss = blip_outputs.loss

                    # Get vision features
                    vision_outputs = model.vision_model(pixel_values)
                    image_features = vision_outputs.pooler_output

                    # Forward pass through report generator
                    gen_loss, logits = report_generator(
                        input_embeddings=image_features,
                        labels=labels,
                        target_ids=input_ids
                    )

                    # Calculate total loss
                    total_loss = gen_loss * 0.5 + blip_loss

                    # Backward pass
                    generator_optimizer.zero_grad()
                    model_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(report_generator.parameters(), max_norm=1.0)
                    generator_optimizer.step()
                    model_optimizer.step()

                    # Track losses
                    train_losses.append(total_loss.item())
                    blip_losses.append(blip_loss.item())
                    gen_losses.append(gen_loss.item())

                    # Update progress bar
                    progress_bar.set_postfix({
                        'Loss': f'{total_loss.item():.4f}',
                        'BLIP': f'{blip_loss.item():.4f}',
                        'Gen': f'{gen_loss.item():.4f}'
                    })

                    # Validate every 200 batches
                    if (batch_idx + 1) % 200 == 0:
                        val_loss, val_blip_loss, val_gen_loss = validate_model()
                        logger.info(f"\nValidation at batch {batch_idx + 1}:")
                        logger.info(f"Val Loss: {val_loss:.4f}")
                        logger.info(f"Val BLIP Loss: {val_blip_loss:.4f}")
                        logger.info(f"Val Gen Loss: {val_gen_loss:.4f}")

                        # Save best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save({
                                'epoch': epoch,
                                'batch': batch_idx,
                                'model_state_dict': model.state_dict(),
                                'report_generator_state_dict': report_generator.state_dict(),
                                'val_loss': val_loss
                            }, 'best_model.pt')

                        # Return to training mode
                        model.train()
                        report_generator.train()

                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    continue

            # Calculate epoch averages
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_blip_loss = sum(blip_losses) / len(blip_losses)
            avg_gen_loss = sum(gen_losses) / len(gen_losses)

            # Validate at end of epoch
            val_loss, val_blip_loss, val_gen_loss = validate_model()

            # Log epoch stats
            logger.info(f"\nEpoch {epoch + 1} Stats:")
            logger.info(f"Training Loss: {avg_train_loss:.4f}")
            logger.info(f"Training BLIP Loss: {avg_blip_loss:.4f}")
            logger.info(f"Training Gen Loss: {avg_gen_loss:.4f}")
            logger.info(f"Validation Loss: {val_loss:.4f}")
            logger.info(f"Validation BLIP Loss: {val_blip_loss:.4f}")
            logger.info(f"Validation Gen Loss: {val_gen_loss:.4f}")

            # Update schedulers
            generator_scheduler.step(val_gen_loss)
            model_scheduler.step(val_blip_loss)

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'report_generator_state_dict': report_generator.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss
            }, f'checkpoint_epoch_{epoch + 1}.pt')

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    train_model(
        "../Data/final_cleaned.csv",
        "/home/ubuntu/new_code/CheXbert/src/datasets/labeled_reports_with_images.csv",
        "/home/ubuntu/new_code/CheXbert/src/image_classifier/split_data/images",
        num_epochs=20,
        batch_size=8,
        num_trainable_layers=2,
        validation_split=0.1
    )