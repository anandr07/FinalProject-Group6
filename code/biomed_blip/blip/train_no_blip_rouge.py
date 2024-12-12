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
from rouge_score import rouge_scorer


class ChestXrayDataset(Dataset):
    def __init__(self, report_df, labels_df, image_dir, processor, tokenizer):
        super().__init__()
        self.image_dir = image_dir
        self.processor = processor
        self.tokenizer = tokenizer

        self.label_columns = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]

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

        image = Image.open(os.path.join(self.image_dir, row['image_id'])).convert('RGB')
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        encoding = self.tokenizer(
            findings,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )

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

    # Unfreeze last few layers of language model
    decoder_layers = model.language_model.model.decoder.layers
    for layer in decoder_layers[-num_trainable_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Keep projection layers trainable
    if hasattr(model, 'language_projection'):
        for param in model.language_projection.parameters():
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")


def evaluate_model(model, data_loader, blip_model, scorer, device, logger):
    """
    Evaluate the model on validation set
    """
    model.eval()
    val_losses = []
    val_rouge_scores = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            vision_outputs = blip_model.vision_model(pixel_values)
            image_features = vision_outputs.pooler_output

            loss, _ = model(
                input_embeddings=image_features,
                target_ids=input_ids,
                labels=labels
            )
            val_losses.append(loss.item())

            generated_reports = model.generate_report(
                input_embeddings=image_features,
                labels=labels
            )

            batch_rouge_scores = []
            for gen_report, ref_report in zip(generated_reports, batch['text']):
                rouge_score = scorer.score(ref_report, gen_report)
                batch_rouge_scores.append(rouge_score['rougeL'].fmeasure)
            val_rouge_scores.extend(batch_rouge_scores)

    avg_val_loss = np.mean(val_losses)
    avg_val_rouge = np.mean(val_rouge_scores)

    logger.info(f"\nValidation Results:")
    logger.info(f"Average Loss: {avg_val_loss:.4f}")
    logger.info(f"Average ROUGE-L: {avg_val_rouge:.4f}")

    return avg_val_loss, avg_val_rouge


def train_model(report_csv_path: str, labels_csv_path: str, image_dir: str, num_epochs: int = 10,
                batch_size: int = 4, validation_split: float = 0.1):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float32,
            device_map='auto',
            max_memory={0: "12GB"}
        )
        model.eval()

        vision_hidden_size = model.vision_model.config.hidden_size
        report_generator = MedicalReportGenerator(input_embedding_dim=vision_hidden_size)
        report_generator = report_generator.to(device)

        vocab_size = report_generator.tokenizer.vocab_size
        logger.info(f"BioGPT vocabulary size: {vocab_size}")

        full_dataset = ChestXrayDataset(
            pd.read_csv(report_csv_path),
            pd.read_csv(labels_csv_path),
            image_dir,
            processor,
            report_generator.tokenizer
        )

        dataset_size = len(full_dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

        optimizer = torch.optim.AdamW(report_generator.parameters(), lr=2e-5)

        best_val_rouge = 0.0
        for epoch in range(num_epochs):
            report_generator.train()
            epoch_losses = []
            epoch_rouge_scores = []
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                                desc=f'Epoch {epoch + 1}/{num_epochs}')

            for batch_idx, batch in progress_bar:
                try:
                    pixel_values = batch['pixel_values'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)

                    with torch.no_grad():
                        vision_outputs = model.vision_model(pixel_values)
                        image_features = vision_outputs.pooler_output

                    loss, _ = report_generator(
                        input_embeddings=image_features,
                        target_ids=input_ids,
                        labels=labels
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(report_generator.parameters(), max_norm=1.0)
                    optimizer.step()

                    current_loss = loss.item()
                    epoch_losses.append(current_loss)
                    avg_loss = sum(epoch_losses) / len(epoch_losses)

                    if batch_idx % 100 == 0:
                        report_generator.eval()
                        with torch.no_grad():
                            sample_report = report_generator.generate_report(
                                input_embeddings=image_features[0:1],
                                labels=labels[0:1]
                            )[0]

                            rouge_score = scorer.score(batch['text'][0], sample_report)
                            rouge_l = rouge_score['rougeL'].fmeasure
                            epoch_rouge_scores.append(rouge_l)

                            logger.info(f"\nSample at batch {batch_idx}:")
                            logger.info(f"Generated: {sample_report}")
                            logger.info(f"Original: {batch['text'][0]}")
                            logger.info(f"ROUGE-L score: {rouge_l:.4f}\n")
                        report_generator.train()

                    progress_bar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Avg Loss': f'{avg_loss:.4f}',
                        'Avg ROUGE-L': f'{np.mean(epoch_rouge_scores):.4f}' if epoch_rouge_scores else 'N/A'
                    })

                except Exception as e:
                    logger.error(f"Batch {batch_idx} error: {str(e)}", exc_info=True)
                    continue

            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            avg_epoch_rouge = np.mean(epoch_rouge_scores) if epoch_rouge_scores else 0.0

            logger.info("\nRunning validation...")
            val_loss, val_rouge = evaluate_model(
                report_generator, val_loader, model, scorer, device, logger
            )

            checkpoint_path = f'checkpoint_epoch_{epoch + 1}.pt'
            is_best = val_rouge > best_val_rouge
            best_val_rouge = max(val_rouge, best_val_rouge)

            torch.save({
                'epoch': epoch,
                'model_state_dict': report_generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_epoch_loss,
                'train_rouge_l': avg_epoch_rouge,
                'val_loss': val_loss,
                'val_rouge_l': val_rouge,
                'is_best': is_best
            }, checkpoint_path)

            if is_best:
                best_checkpoint_path = 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': report_generator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_rouge_l': val_rouge
                }, best_checkpoint_path)
                logger.info(f"New best model saved with validation ROUGE-L: {val_rouge:.4f}")

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