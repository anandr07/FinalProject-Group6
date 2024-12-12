import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import List


class MedicalReportGenerator(nn.Module):
    def __init__(self, input_embedding_dim: int, num_labels: int = 14):
        super().__init__()
        self.base_model_name = 'microsoft/BioGPT'
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        self.model.gradient_checkpointing_enable()

        # Update PEFT configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)

        # Set special tokens
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.bos_token = self.tokenizer.eos_token

        # Store token IDs for easy access
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        # Input and label projections
        self.input_projection = nn.Sequential(
            nn.Linear(input_embedding_dim, self.model.config.hidden_size),
            nn.LayerNorm(self.model.config.hidden_size),
            nn.Dropout(0.1)
        )

        self.label_projection = nn.Sequential(
            nn.Linear(num_labels, self.model.config.hidden_size),
            nn.LayerNorm(self.model.config.hidden_size),
            nn.Dropout(0.1)
        )

    def _validate_input_ids(self, input_ids):
        """Validate and fix input IDs if needed"""
        vocab_size = self.tokenizer.vocab_size
        if torch.any(input_ids >= vocab_size):
            print(f"Warning: input_ids contains values >= vocab_size ({vocab_size})")
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        return input_ids

    def forward(self, input_embeddings: torch.Tensor, target_ids: torch.Tensor = None, labels: torch.Tensor = None):
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device

        try:
            # Project input embeddings
            projected_img_embeddings = self.input_projection(input_embeddings)
            projected_img_embeddings = projected_img_embeddings.unsqueeze(1)

            if target_ids is not None and labels is not None:
                # Validate target_ids
                target_ids = self._validate_input_ids(target_ids)

                # Project label embeddings
                projected_label_embeddings = self.label_projection(labels)
                projected_label_embeddings = projected_label_embeddings.unsqueeze(1)

                # Get token embeddings with proper handling
                token_embeddings = self.model.get_input_embeddings()(target_ids)

                # Concatenate embeddings
                combined_embeddings = torch.cat([
                    projected_img_embeddings,
                    projected_label_embeddings,
                    token_embeddings
                ], dim=1)

                # Create attention mask
                attention_mask = torch.ones(
                    combined_embeddings.size()[:2],
                    dtype=torch.long,
                    device=device
                )

                # Create labels for loss calculation
                padding = torch.full(
                    (batch_size, 2),  # 2 for image and label embeddings
                    -100,  # Ignore index for loss calculation
                    dtype=torch.long,
                    device=device
                )
                shifted_labels = torch.cat([padding, target_ids], dim=1)

                # Forward pass
                outputs = self.model(
                    inputs_embeds=combined_embeddings,
                    attention_mask=attention_mask,
                    labels=shifted_labels,
                    return_dict=True
                )

                return outputs.loss, outputs.logits
            else:
                raise ValueError("Both target_ids and labels must be provided during training")

        except Exception as e:
            print(f"Forward pass error: {str(e)}")
            raise

    def generate_report(self, input_embeddings: torch.Tensor, labels: torch.Tensor, max_length: int = 150) -> List[str]:
        self.eval()
        device = input_embeddings.device

        try:
            # Project embeddings
            projected_img_embeddings = self.input_projection(input_embeddings)
            projected_img_embeddings = projected_img_embeddings.unsqueeze(1)

            projected_label_embeddings = self.label_projection(labels)
            projected_label_embeddings = projected_label_embeddings.unsqueeze(1)

            # Combine embeddings
            combined_embeddings = torch.cat([
                projected_img_embeddings,
                projected_label_embeddings
            ], dim=1)

            # Create attention mask
            attention_mask = torch.ones(
                combined_embeddings.size()[:2],
                dtype=torch.long,
                device=device
            )

            # Add generation config
            gen_config = {
                'max_length': max_length,
                'min_length': 30,
                'num_return_sequences': 1,
                'do_sample': True,
                'top_k': 50,
                'top_p': 0.9,
                'temperature': 0.7,
                'repetition_penalty': 1.2,
                'length_penalty': 1.0,
                'no_repeat_ngram_size': 3,
                'pad_token_id': self.pad_token_id,
                'eos_token_id': self.eos_token_id,
                'use_cache': True
            }

            # Generate text
            outputs = self.model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=attention_mask,
                **gen_config
            )

            # Decode outputs
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts = [text.strip() for text in generated_texts]
            generated_texts = [text if text else "No abnormalities detected." for text in generated_texts]

            return generated_texts

        except Exception as e:
            print(f"Generation error: {str(e)}")
            return ["Error generating report"]

    def generate_reports_batch(self, input_embeddings, labels=None, max_length=512, num_beams=4):
        """
        Generate reports for a batch of images at once
        """
        batch_size = input_embeddings.size(0)

        # Project embeddings
        projected_embeddings = self.input_projection(input_embeddings)

        # Generate tokens
        outputs = self.model.generate(
            inputs_embeds=projected_embeddings,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            early_stopping=True
        )

        # Decode all outputs at once
        reports = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return reports

    @property
    def vocab_size(self):
        """Get vocabulary size of the tokenizer"""
        return len(self.tokenizer)