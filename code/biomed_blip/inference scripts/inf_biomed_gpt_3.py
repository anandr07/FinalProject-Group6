import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import open_clip
from PIL import Image
import logging
from pathlib import Path
from typing import List, Dict, Union
import argparse


class MedicalReportGenerator(nn.Module):
    def __init__(self, input_embedding_dim: int, num_labels: int = 14):
        super().__init__()
        self.base_model_name = 'microsoft/BioGPT'
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)

        # Update PEFT configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_embedding_dim, self.model.config.hidden_size),
            nn.LayerNorm(self.model.config.hidden_size),
            nn.Dropout(0.1)
        )

        # Label projection
        self.label_projection = nn.Sequential(
            nn.Linear(num_labels, self.model.config.hidden_size),
            nn.LayerNorm(self.model.config.hidden_size),
            nn.Dropout(0.1)
        )

        # Set special tokens
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

    def generate_report(self, input_embeddings: torch.Tensor, labels: torch.Tensor, max_length: int = 150) -> List[str]:
        self.model.eval()

        try:
            # Project embeddings
            projected_img_embeddings = self.input_projection(input_embeddings)
            projected_img_embeddings = projected_img_embeddings.unsqueeze(1)

            projected_label_embeddings = self.label_projection(labels)
            projected_label_embeddings = projected_label_embeddings.unsqueeze(1)

            combined_embeddings = torch.cat(
                [projected_img_embeddings, projected_label_embeddings],
                dim=1
            )

            # Create attention mask
            attention_mask = torch.ones(
                combined_embeddings.size()[:2],
                device=input_embeddings.device,
                dtype=torch.long
            )

            # Generate with parameters
            outputs = self.model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=30,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.5,
                temperature=0.3,
                repetition_penalty=1.2,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts = [text.strip() for text in generated_texts]
            generated_texts = [text if text else "No abnormalities detected." for text in generated_texts]

            return generated_texts

        except Exception as e:
            print(f"Error in generate_report: {str(e)}")
            return ["Error generating report"]


class TrainableMedicalReportInference:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.logger = self._setup_logger()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Initialize BiomedCLIP
        self.model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name)
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        # Move models to device
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

        # Initialize report generator
        self.report_generator = MedicalReportGenerator(input_embedding_dim=512)
        self.report_generator = self.report_generator.to(self.device)
        self.report_generator.eval()

        # Load checkpoint
        self._load_checkpoint(checkpoint_path)

        self.label_names = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]

    def _setup_logger(self) -> logging.Logger:
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _load_checkpoint(self, checkpoint_path: str):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.clip_model.load_state_dict(checkpoint['model_state_dict'])
            self.report_generator.load_state_dict(checkpoint['report_generator_state_dict'])
            self.logger.info("Successfully loaded checkpoint")
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    def process_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert('RGB')
            processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
            return processed_image
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            raise

    def generate_report_with_similarity(self,
                                        image_path: Union[str, Path],
                                        reference_text: str = None,
                                        labels: Dict[str, float] = None) -> Dict[str, Union[str, float]]:
        """
        Generate a report and optionally calculate similarity with reference text
        """
        try:
            # Process image
            processed_image = self.process_image(image_path)

            # Get image features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(processed_image)
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

                # Calculate similarity if reference text is provided
                similarity_score = None
                if reference_text:
                    text_features = self.clip_model.encode_text(
                        self.tokenizer(reference_text).to(self.device)
                    )
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    similarity_score = (image_features @ text_features.t()).item()

            # Prepare labels tensor
            if labels is None:
                labels = {name: 0.0 for name in self.label_names}
            label_tensor = torch.tensor(
                [[labels.get(name, 0.0) for name in self.label_names]],
                dtype=torch.float,
                device=self.device
            )

            # Generate report
            generated_report = self.report_generator.generate_report(
                input_embeddings=image_features,
                labels=label_tensor
            )[0]

            return {
                'report': generated_report,
                'similarity_score': similarity_score if similarity_score is not None else "No reference text provided"
            }

        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {
                'report': "Error generating medical report.",
                'similarity_score': None
            }


def main():
    parser = argparse.ArgumentParser(description='Generate medical reports from chest X-ray images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--reference', type=str, help='Reference text to calculate similarity score')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--labels', type=str, help='JSON string of labels and their values')

    args = parser.parse_args()

    # Initialize inference
    inferencer = TrainableMedicalReportInference(args.checkpoint, args.device)

    # Parse labels if provided
    labels = None
    if args.labels:
        import json
        labels = json.loads(args.labels)

    # Generate report and get similarity score
    result = inferencer.generate_report_with_similarity(
        args.image,
        args.reference,
        labels
    )

    print("\nGenerated Report:")
    print("-" * 50)
    print(result['report'])
    print("-" * 50)
    print(f"Similarity Score: {result['similarity_score']}")


if __name__ == "__main__":
    main()