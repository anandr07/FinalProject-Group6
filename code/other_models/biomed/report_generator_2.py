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

        # Update input projection to match dimensions
        # input_embedding_dim is 512 from CLIP, and model.config.hidden_size is the BioGPT dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_embedding_dim, self.model.config.hidden_size),
            nn.LayerNorm(self.model.config.hidden_size),
            nn.Dropout(0.1)
        )

        # Set special tokens
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

            # Add label embedding layer
        self.label_projection = nn.Sequential(
            nn.Linear(num_labels, self.model.config.hidden_size),
            nn.LayerNorm(self.model.config.hidden_size),
            nn.Dropout(0.1)
        )

    # def forward(self, input_embeddings: torch.Tensor, labels: torch.Tensor, target_ids: torch.Tensor = None):
    #     batch_size = input_embeddings.size(0)
    #
    #     # Project image embeddings
    #     projected_img_embeddings = self.input_projection(input_embeddings)
    #     projected_img_embeddings = projected_img_embeddings.unsqueeze(1)
    #
    #     # Project label embeddings
    #     projected_label_embeddings = self.label_projection(labels)
    #     projected_label_embeddings = projected_label_embeddings.unsqueeze(1)
    #
    #     # Concatenate image and label embeddings
    #     combined_embeddings = torch.cat(
    #         [projected_img_embeddings, projected_label_embeddings],
    #         dim=1
    #     )
    #
    #     if target_ids is not None:
    #         token_embeddings = self.model.get_input_embeddings()(target_ids)
    #         inputs_embeds = torch.cat([combined_embeddings, token_embeddings], dim=1)
    #
    #         attention_mask = torch.ones(
    #             inputs_embeds.size()[:2],
    #             device=input_embeddings.device,
    #             dtype=torch.long
    #         )
    #
    #         # Adjust labels
    #         padding = torch.full(
    #             (batch_size, 2),  # 2 for image and label embeddings
    #             -100,
    #             dtype=torch.long,
    #             device=target_ids.device
    #         )
    #         labels = torch.cat([padding, target_ids], dim=1)
    #
    #         outputs = self.model(
    #             inputs_embeds=inputs_embeds,
    #             attention_mask=attention_mask,
    #             labels=labels,
    #             return_dict=True
    #         )
    #         return outputs.loss, outputs.logits

    def forward(self, input_embeddings: torch.Tensor, target_ids: torch.Tensor = None, labels: torch.Tensor = None):
        # input_embeddings shape: (batch_size, 512)
        batch_size = input_embeddings.size(0)

        # Project input embeddings to model's hidden size
        projected_img_embeddings = self.input_projection(input_embeddings)  # (batch_size, hidden_size)
        projected_img_embeddings = projected_img_embeddings.unsqueeze(1)  # (batch_size, 1, hidden_size)

        if target_ids is not None:
            # Get token embeddings for the target sequence
            token_embeddings = self.model.get_input_embeddings()(target_ids)

            # Process labels if provided
            if labels is not None:
                projected_label_embeddings = self.label_projection(labels)
                projected_label_embeddings = projected_label_embeddings.unsqueeze(1)
                # Concatenate image and label embeddings
                combined_embeddings = torch.cat([projected_img_embeddings, projected_label_embeddings], dim=1)
            else:
                combined_embeddings = projected_img_embeddings

            # Concatenate with token embeddings
            inputs_embeds = torch.cat([combined_embeddings, token_embeddings], dim=1)

            # Create attention mask
            attention_mask = torch.ones(
                inputs_embeds.size()[:2],
                device=input_embeddings.device,
                dtype=torch.long
            )

            # Prepare labels (shift right)
            padding_length = 2 if labels is not None else 1  # Extra position for label embedding if used
            padding = torch.full(
                (batch_size, padding_length),
                -100,
                dtype=torch.long,
                device=target_ids.device
            )
            shifted_labels = torch.cat([padding, target_ids], dim=1)

            # Forward pass
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=shifted_labels,
                return_dict=True
            )
            return outputs.loss, outputs.logits
        else:
            raise ValueError("Target IDs must be provided during training.")

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
    # def forward(self, input_embeddings: torch.Tensor, target_ids: torch.Tensor = None):
    #     # input_embeddings shape: (batch_size, 512)
    #     batch_size = input_embeddings.size(0)
    #
    #     # Project input embeddings to model's hidden size
    #     projected_embeddings = self.input_projection(input_embeddings)  # (batch_size, hidden_size)
    #     projected_embeddings = projected_embeddings.unsqueeze(1)  # (batch_size, 1, hidden_size)
    #
    #     if target_ids is not None:
    #         # Get token embeddings for the target sequence
    #         token_embeddings = self.model.get_input_embeddings()(target_ids)
    #
    #         # Concatenate projected embeddings with token embeddings
    #         inputs_embeds = torch.cat([projected_embeddings, token_embeddings], dim=1)
    #
    #         # Create attention mask
    #         attention_mask = torch.ones(
    #             inputs_embeds.size()[:2],
    #             device=input_embeddings.device,
    #             dtype=torch.long
    #         )
    #
    #         # Prepare labels (shift right)
    #         labels = torch.full(
    #             (batch_size, 1),
    #             -100,
    #             dtype=torch.long,
    #             device=target_ids.device
    #         )
    #         labels = torch.cat([labels, target_ids], dim=1)
    #
    #         # Forward pass
    #         outputs = self.model(
    #             inputs_embeds=inputs_embeds,
    #             attention_mask=attention_mask,
    #             labels=labels,
    #             return_dict=True
    #         )
    #         return outputs.loss, outputs.logits
    #     else:
    #         raise ValueError("Target IDs must be provided during training.")
    #
    # def generate_report(self, input_embeddings: torch.Tensor, max_length: int = 150) -> List[str]:
    #     self.model.eval()  # Ensure model is in evaluation mode
    #
    #     try:
    #         # Project input embeddings
    #         projected_embeddings = self.input_projection(input_embeddings)
    #         projected_embeddings = projected_embeddings.unsqueeze(1)
    #
    #         # Get BOS token embedding
    #         bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
    #         bos_embedding = self.model.get_input_embeddings()(
    #             torch.tensor([[bos_token_id]], device=input_embeddings.device))
    #
    #         # Combine embeddings
    #         inputs_embeds = torch.cat([bos_embedding, projected_embeddings], dim=1)
    #
    #         # Create attention mask
    #         attention_mask = torch.ones(inputs_embeds.size()[:2], device=input_embeddings.device, dtype=torch.long)
    #
    #         # Generate with more controlled parameters
    #         outputs = self.model.generate(
    #             inputs_embeds=inputs_embeds,
    #             attention_mask=attention_mask,
    #             max_length=max_length,
    #             min_length=30,  # Add minimum length
    #             num_return_sequences=1,
    #             do_sample=True,
    #             top_k=50,
    #             top_p=0.95,
    #             temperature=0.7,
    #             repetition_penalty=1.2,  # Add repetition penalty
    #             length_penalty=1.0,  # Add length penalty
    #             no_repeat_ngram_size=3,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             eos_token_id=self.tokenizer.eos_token_id,
    #             use_cache=True
    #         )
    #
    #         # Decode outputs
    #         generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #
    #         # Filter out empty generations
    #         generated_texts = [text.strip() for text in generated_texts]
    #         generated_texts = [text if text else "No abnormalities detected." for text in generated_texts]
    #
    #         return generated_texts
    #
    #     except Exception as e:
    #         print(f"Error in generate_report: {str(e)}")
    #         return ["Error generating report"]