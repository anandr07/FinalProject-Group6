import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from pathlib import Path
from typing import Union, List
import logging
from other_models.report_generator_bioclip import MedicalReportGenerator
import sys
import os
import pandas as pd
import numpy as np


class ChestXrayReportGenerator:
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Initialize the chest X-ray report generator
        """
        self.logger = self._setup_logger()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        self.label_columns = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]

        try:
            # Load BLIP2 model and processor
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float32,
                device_map='auto',
                max_memory={0: "12GB"}
            )
            self.blip_model.eval()

            # Initialize report generator
            vision_hidden_size = self.blip_model.vision_model.config.hidden_size
            self.report_generator = MedicalReportGenerator(input_embedding_dim=vision_hidden_size)

            # Load trained weights
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.report_generator.load_state_dict(checkpoint['model_state_dict'])
            self.report_generator.to(self.device)
            self.report_generator.eval()

            self.logger.info("Models loaded successfully")

        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration"""
        logger = logging.getLogger('ChestXrayReportGenerator')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)

        return logger

    def _validate_and_process_label(self, value):
        """Validate and process a single label value"""
        try:
            # Convert to float and handle NaN
            val = float(value) if not pd.isna(value) else 0.0
            # Clip to valid range [0, 1]
            return np.clip(val, 0.0, 1.0)
        except:
            return 0.0

    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess a single image for the model
        """
        try:
            image = Image.open(image_path).convert('RGB')
            processed = self.processor(images=image, return_tensors="pt")
            return processed.pixel_values.to(self.device)
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise

    def generate_report(self, image_path: Union[str, Path], labels_df: pd.DataFrame) -> dict:
        """
        Generate a medical report for a given chest X-ray image
        """
        try:
            self.logger.info(f"Generating report for image: {image_path}")
            image_id = Path(image_path).stem

            # Get labels for this image
            image_labels = labels_df[labels_df['image_id'] == f"{image_id}.png"]
            if len(image_labels) == 0:
                raise ValueError(f"No labels found for image {image_id}")

            # Process and validate labels
            label_values = []
            for col in self.label_columns:
                if col in image_labels.columns:
                    val = self._validate_and_process_label(image_labels[col].iloc[0])
                else:
                    val = 0.0
                label_values.append(val)

            # Create labels tensor
            labels = torch.tensor(label_values, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Verify tensor values
            if torch.any(torch.isnan(labels)) or torch.any(torch.isinf(labels)):
                raise ValueError("Labels contain NaN or Inf values")
            if torch.any(labels < 0) or torch.any(labels > 1):
                raise ValueError("Labels contain values outside [0,1] range")

            # Debug info
            self.logger.info(f"Label values: {label_values}")
            self.logger.info(f"Label tensor shape: {labels.shape}")

            # Preprocess image
            pixel_values = self.preprocess_image(image_path)

            # Generate image features using BLIP2
            with torch.no_grad():
                vision_outputs = self.blip_model.vision_model(pixel_values)
                image_features = vision_outputs.pooler_output

                # Generate report
                generated_report = self.report_generator.generate_report(
                    input_embeddings=image_features,
                    labels=labels
                )[0]

            # Create labels dictionary
            labels_dict = {
                label: float(val) for label, val in zip(self.label_columns, label_values)
            }

            return {
                'report': generated_report,
                'labels': labels_dict
            }

        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise


def main():
    # Define paths
    base_dir = "/home/ubuntu/new_code"
    checkpoint_path = os.path.join(base_dir, "other_models/checkpoint_blip_4/checkpoint_epoch_20.pt")
    image_path = os.path.join(base_dir, "CheXbert/src/image_classifier/split_data/images/CXR1_1_IM-0001-3001.png")
    labels_path = os.path.join(base_dir, "CheXbert/src/datasets/labeled_reports_with_images.csv")
    output_dir = os.path.join(base_dir, "other_models/blip/checkpoints_7k/generated_reports")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load labels
    labels_df = pd.read_csv(labels_path)
    print("Label columns in DataFrame:", labels_df.columns.tolist())

    # Initialize the generator
    generator = ChestXrayReportGenerator(checkpoint_path)

    result = generator.generate_report(image_path, labels_df)
    output_file = Path(output_dir) / f"{Path(image_path).stem}_report.txt"

    with open(output_file, 'w') as f:
        f.write(f"Report:\n{result['report']}\n\nLabels:\n")
        for label, value in result['labels'].items():
            f.write(f"{label}: {value:.3f}\n")

    print(f"Generated report for {Path(image_path).name}")
    print("\nGenerated Report:")
    print(result['report'])
    print("\nLabels:")
    for label, value in result['labels'].items():
        print(f"{label}: {value:.3f}")


if __name__ == "__main__":
    main()