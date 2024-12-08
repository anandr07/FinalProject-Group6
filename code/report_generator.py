
# report_generator.py - working biogpt
# report_generator.py
# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import get_peft_model, LoraConfig, TaskType
# from typing import List
#
# class MedicalReportGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Use BioGPT as the base model
#         self.base_model_name = 'microsoft/biogpt'
#         self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
#         self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing during training
#
#         # PEFT configuration with target_modules specified
#         peft_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             inference_mode=False,
#             r=16,
#             lora_alpha=32,
#             lora_dropout=0.1,
#             target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
#         )
#         self.model = get_peft_model(self.model, peft_config)
#
#         # Projection layer to map image embeddings to model's embedding size
#         self.input_projection = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
#
#         # Ensure special tokens are set
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         if self.tokenizer.bos_token_id is None:
#             self.tokenizer.bos_token = self.tokenizer.eos_token
#         if self.tokenizer.eos_token_id is None:
#             self.tokenizer.eos_token = '</s>'
#             self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
#         self.model.config.pad_token_id = self.tokenizer.pad_token_id
#         self.model.config.bos_token_id = self.tokenizer.bos_token_id
#         self.model.config.eos_token_id = self.tokenizer.eos_token_id
#
#     def forward(self, input_embeddings: torch.Tensor, target_ids: torch.Tensor = None):
#         # Project input embeddings to model's hidden size
#         projected_embeddings = self.input_projection(input_embeddings)
#         projected_embeddings = projected_embeddings.unsqueeze(1)  # Add sequence dimension
#
#         # Prepare inputs for the model
#         if target_ids is not None:
#             # Get token embeddings for the target sequence
#             token_embeddings = self.model.get_input_embeddings()(target_ids)
#             # Concatenate projected image embeddings with token embeddings
#             inputs_embeds = torch.cat([projected_embeddings, token_embeddings], dim=1)
#             # Adjust attention mask
#             attention_mask = torch.ones(inputs_embeds.size()[:2], device=input_embeddings.device, dtype=torch.long)
#             # Pad labels with -100 at the beginning to match input length
#             padding = torch.full((target_ids.size(0), 1), -100, dtype=torch.long, device=target_ids.device)
#             labels = torch.cat([padding, target_ids], dim=1)
#             # Forward pass with labels
#             outputs = self.model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 labels=labels,
#                 return_dict=True
#             )
#             return outputs.loss, outputs.logits
#         else:
#             raise ValueError("Target IDs must be provided during training.")
#
#     def generate_report(self, input_embeddings: torch.Tensor, max_length: int = 150) -> List[str]:
#         # Temporarily disable gradient checkpointing
#         self.model.gradient_checkpointing_disable()
#         # Project input embeddings to model's hidden size
#         projected_embeddings = self.input_projection(input_embeddings)
#         projected_embeddings = projected_embeddings.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
#
#         # Get BOS token id
#         bos_token_id = self.tokenizer.bos_token_id
#         if bos_token_id is None:
#             raise ValueError("bos_token_id is not set in the tokenizer.")
#
#         # Get embedding of BOS token
#         bos_embedding = self.model.get_input_embeddings()(torch.tensor([[bos_token_id]]).to(input_embeddings.device))
#         # Shape: (1, 1, hidden_size)
#
#         # Repeat bos_embedding for batch size
#         bos_embedding = bos_embedding.expand(input_embeddings.size(0), -1, -1)  # Shape: (batch_size, 1, hidden_size)
#
#         # Concatenate bos_embedding and projected_embeddings
#         inputs_embeds = torch.cat([bos_embedding, projected_embeddings], dim=1)  # Shape: (batch_size, 2, hidden_size)
#
#         # Create attention mask
#         batch_size = inputs_embeds.size(0)
#         attention_mask = torch.ones((batch_size, inputs_embeds.size(1)), device=inputs_embeds.device, dtype=torch.long)
#
#         # Generate text
#         outputs = self.model.generate(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             max_length=max_length,
#             min_length=10,
#             num_return_sequences=1,
#             do_sample=True,
#             top_k=50,
#             top_p=0.85,
#             temperature=0.8,
#             length_penalty=1.0,
#             repetition_penalty=1.2,
#             no_repeat_ngram_size=3,
#             pad_token_id=self.tokenizer.pad_token_id,
#             eos_token_id=self.tokenizer.eos_token_id,
#             use_cache=True,  # Ensure use_cache is True during generation
#         )
#         # Re-enable gradient checkpointing
#         self.model.gradient_checkpointing_enable()
#         generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         return generated_texts

# Sample inference parameters
# outputs = self.model.generate(
#                 inputs_embeds=combined_embeddings,
#                 attention_mask=attention_mask,
#                 max_length=combined_embeddings.size(1) + max_length,  # adjust max_length accordingly
#                 min_length=10,
#                 num_return_sequences=1,
#                 do_sample=True,
#                 top_k=50,
#                 top_p=0.9,
#                 temperature=0.8,
#                 no_repeat_ngram_size=3,
#                 length_penalty=1.0,
#                 repetition_penalty=1.0,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#                 use_cache=True,

# Trail-Cite

# report_generator.py
#
# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import get_peft_model, LoraConfig, TaskType
# from typing import List
#
# class MedicalReportGenerator(nn.Module):
#   def __init__(self, input_embedding_dim: int):
#       super().__init__()
#       # Use BioGPT as the base model
#       self.base_model_name = 'microsoft/BioGPT'
#       self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
#       self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
#       self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing during training
#
#       # PEFT configuration with target_modules specified
#       peft_config = LoraConfig(
#           task_type=TaskType.CAUSAL_LM,
#           inference_mode=False,
#           r=16,
#           lora_alpha=32,
#           lora_dropout=0.1,
#           target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
#       )
#       self.model = get_peft_model(self.model, peft_config)
#
#       # Projection layer to map input embeddings to model's embedding size
#       self.input_projection = nn.Linear(input_embedding_dim, self.model.config.hidden_size)
#
#       # Ensure special tokens are set
#       if self.tokenizer.pad_token_id is None:
#           self.tokenizer.pad_token = self.tokenizer.eos_token
#       if self.tokenizer.bos_token_id is None:
#           self.tokenizer.bos_token = self.tokenizer.eos_token
#       if self.tokenizer.eos_token_id is None:
#           self.tokenizer.eos_token = '</s>'
#           self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
#       self.model.config.pad_token_id = self.tokenizer.pad_token_id
#       self.model.config.bos_token_id = self.tokenizer.bos_token_id
#       self.model.config.eos_token_id = self.tokenizer.eos_token_id
#
#   def forward(self, input_embeddings: torch.Tensor, target_ids: torch.Tensor = None):
#       # Project input embeddings to model's hidden size
#       projected_embeddings = self.input_projection(input_embeddings)
#       projected_embeddings = projected_embeddings.unsqueeze(1)  # Add sequence dimension
#
#       # Prepare inputs for the model
#       if target_ids is not None:
#           # Get token embeddings for the target sequence
#           token_embeddings = self.model.get_input_embeddings()(target_ids)
#           # Concatenate projected embeddings with token embeddings
#           inputs_embeds = torch.cat([projected_embeddings, token_embeddings], dim=1)
#           # Adjust attention mask
#           attention_mask = torch.ones(inputs_embeds.size()[:2], device=input_embeddings.device, dtype=torch.long)
#           # Pad labels with -100 at the beginning to match input length
#           padding = torch.full((target_ids.size(0), 1), -100, dtype=torch.long, device=target_ids.device)
#           labels = torch.cat([padding, target_ids], dim=1)
#           # Forward pass with labels
#           outputs = self.model(
#               inputs_embeds=inputs_embeds,
#               attention_mask=attention_mask,
#               labels=labels,
#               return_dict=True
#           )
#           return outputs.loss, outputs.logits
#       else:
#           raise ValueError("Target IDs must be provided during training.")
#
#   def generate_report(self, input_embeddings: torch.Tensor, max_length: int = 150) -> List[str]:
#       # Temporarily disable gradient checkpointing
#       self.model.gradient_checkpointing_disable()
#       # Project input embeddings to model's hidden size
#       projected_embeddings = self.input_projection(input_embeddings)
#       projected_embeddings = projected_embeddings.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
#
#       # Get BOS token id
#       bos_token_id = self.tokenizer.bos_token_id
#       if bos_token_id is None:
#           raise ValueError("bos_token_id is not set in the tokenizer.")
#
#       # Get embedding of BOS token
#       bos_embedding = self.model.get_input_embeddings()(torch.tensor([[bos_token_id]]).to(input_embeddings.device))
#       # Shape: (1, 1, hidden_size)
#
#       # Repeat bos_embedding for batch size
#       bos_embedding = bos_embedding.expand(input_embeddings.size(0), -1, -1)  # Shape: (batch_size, 1, hidden_size)
#
#       # Concatenate bos_embedding and projected_embeddings
#       inputs_embeds = torch.cat([bos_embedding, projected_embeddings], dim=1)  # Shape: (batch_size, 2, hidden_size)
#
#       # Create attention mask
#       batch_size = inputs_embeds.size(0)
#       attention_mask = torch.ones((batch_size, inputs_embeds.size(1)), device=inputs_embeds.device, dtype=torch.long)
#
#       # Generate text
#       outputs = self.model.generate(
#           inputs_embeds=inputs_embeds,
#           attention_mask=attention_mask,
#           max_length=max_length,
#           num_return_sequences=1,
#           do_sample=True,
#           top_k=50,
#           top_p=0.95,
#           temperature=0.7,
#           pad_token_id=self.tokenizer.pad_token_id,
#           eos_token_id=self.tokenizer.eos_token_id,
#           use_cache=True,  # Ensure use_cache is True during generation
#       )
#       # Re-enable gradient checkpointing
#       self.model.gradient_checkpointing_enable()
#       generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
#       return generated_texts

# Prompt - BioGPT
# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import get_peft_model, LoraConfig, TaskType
# from typing import List
#
# class MedicalReportGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Use BioGPT as the base model
#         self.base_model_name = 'microsoft/biogpt'
#         self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
#         self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing during training
#
#         # PEFT configuration with target_modules specified
#         peft_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             inference_mode=False,
#             r=16,
#             lora_alpha=32,
#             lora_dropout=0.1,
#             target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
#         )
#         self.model = get_peft_model(self.model, peft_config)
#
#         # Projection layer to map image embeddings to model's embedding size
#         self.input_projection = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
#
#         # Ensure special tokens are set
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         if self.tokenizer.bos_token_id is None:
#             self.tokenizer.bos_token = self.tokenizer.eos_token
#         if self.tokenizer.eos_token_id is None:
#             self.tokenizer.eos_token = '</s>'
#             self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
#         self.model.config.pad_token_id = self.tokenizer.pad_token_id
#         self.model.config.bos_token_id = self.tokenizer.bos_token_id
#         self.model.config.eos_token_id = self.tokenizer.eos_token_id
#
#     def forward(self, input_embeddings: torch.Tensor, input_ids: torch.Tensor, target_ids: torch.Tensor = None):
#         # Project input embeddings to model's hidden size
#         projected_embeddings = self.input_projection(input_embeddings)
#         projected_embeddings = projected_embeddings.unsqueeze(1)  # Add sequence dimension
#
#         # Get token embeddings for the input_ids (prompt)
#         token_embeddings = self.model.get_input_embeddings()(input_ids)
#         # Concatenate projected image embeddings with token embeddings
#         inputs_embeds = torch.cat([projected_embeddings, token_embeddings], dim=1)
#         # Adjust attention mask
#         attention_mask = torch.ones(inputs_embeds.size()[:2], device=input_embeddings.device, dtype=torch.long)
#         # Prepare labels
#         if target_ids is not None:
#             # Pad labels with -100 at the beginning to match input length
#             padding = torch.full((target_ids.size(0), projected_embeddings.size(1)), -100, dtype=torch.long, device=target_ids.device)
#             labels = torch.cat([padding, target_ids], dim=1)
#             # Forward pass with labels
#             outputs = self.model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 labels=labels,
#                 return_dict=True
#             )
#             return outputs.loss, outputs.logits
#         else:
#             # For generation
#             outputs = self.model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 return_dict=True
#             )
#             return outputs
#
#     def generate_report(self, input_embeddings: torch.Tensor, input_ids: torch.Tensor, max_length: int = 150) -> List[str]:
#         # Temporarily disable gradient checkpointing
#         self.model.gradient_checkpointing_disable()
#         # Project input embeddings to model's hidden size
#         projected_embeddings = self.input_projection(input_embeddings)
#         projected_embeddings = projected_embeddings.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
#
#         # Get token embeddings for the input_ids (prompt)
#         token_embeddings = self.model.get_input_embeddings()(input_ids)
#         # Concatenate projected image embeddings with token embeddings
#         inputs_embeds = torch.cat([projected_embeddings, token_embeddings], dim=1)
#         # Create attention mask
#         batch_size, seq_length, _ = inputs_embeds.size()
#         attention_mask = torch.ones((batch_size, seq_length), device=input_embeddings.device, dtype=torch.long)
#
#         # Generate text
#         outputs = self.model.generate(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             max_length=seq_length + max_length,
#             num_return_sequences=1,
#             do_sample=True,
#             top_k=30,
#             top_p=0.85,
#             temperature=0.6,
#             pad_token_id=self.tokenizer.pad_token_id,
#             eos_token_id=self.tokenizer.eos_token_id,
#             use_cache=True,  # Ensure use_cache is True during generation
#         )
#         # Re-enable gradient checkpointing
#         self.model.gradient_checkpointing_enable()
#         # Extract the generated text after the prompt
#         generated_texts = []
#         for output in outputs:
#             generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
#             # Remove the prompt from the generated text
#             prompt_length = input_ids.size(1)
#             generated_text = generated_text[prompt_length:]
#             generated_texts.append(generated_text.strip())
#         return generated_texts

# corected prompt - but no output
# report_generator.py
# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import get_peft_model, LoraConfig, TaskType
# from typing import List
# #
# class MedicalReportGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Use BioGPT as the base model
#         self.base_model_name = 'microsoft/biogpt'
#         self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
#         self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing during training
#
#         # PEFT configuration with target_modules specified
#         peft_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             inference_mode=False,
#             r=16,
#             lora_alpha=32,
#             lora_dropout=0.1,
#             target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
#         )
#         self.model = get_peft_model(self.model, peft_config)
#
#         # Projection layer to map image embeddings to model's embedding size
#         self.input_projection = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
#
#         # Ensure special tokens are set
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         if self.tokenizer.bos_token_id is None:
#             self.tokenizer.bos_token = self.tokenizer.eos_token
#         if self.tokenizer.eos_token_id is None:
#             self.tokenizer.eos_token = '</s>'
#             self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
#         self.model.config.pad_token_id = self.tokenizer.pad_token_id
#         self.model.config.bos_token_id = self.tokenizer.bos_token_id
#         self.model.config.eos_token_id = self.tokenizer.eos_token_id
#
#     def forward(self, input_embeddings: torch.Tensor, prompt_input_ids: torch.Tensor, target_ids: torch.Tensor = None):
#         # Project input embeddings to model's hidden size
#         projected_embeddings = self.input_projection(input_embeddings)
#         projected_embeddings = projected_embeddings.unsqueeze(1)  # Add sequence dimension
#
#         # Concatenate prompt_input_ids and target_ids to form the full input_ids
#         if target_ids is not None:
#             full_input_ids = torch.cat([prompt_input_ids, target_ids], dim=1)
#         else:
#             full_input_ids = prompt_input_ids
#
#         # Get token embeddings for the full input_ids
#         token_embeddings = self.model.get_input_embeddings()(full_input_ids)
#
#         # Concatenate projected image embeddings with token embeddings
#         inputs_embeds = torch.cat([projected_embeddings, token_embeddings], dim=1)
#
#         # Adjust attention mask
#         attention_mask = torch.ones(inputs_embeds.size()[:2], device=input_embeddings.device, dtype=torch.long)
#
#         if target_ids is not None:
#             # Create labels with -100 for image and prompt tokens
#             padding_length = projected_embeddings.size(1) + prompt_input_ids.size(1)
#             padding = torch.full((target_ids.size(0), padding_length), -100, dtype=torch.long, device=target_ids.device)
#             labels = torch.cat([padding, target_ids], dim=1)
#
#             # Forward pass with labels
#             outputs = self.model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 labels=labels,
#                 return_dict=True
#             )
#             return outputs.loss, outputs.logits
#         else:
#             # For generation
#             outputs = self.model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 return_dict=True
#             )
#             return outputs
#
#     # def generate_report(self, input_embeddings: torch.Tensor, prompt_input_ids: torch.Tensor, max_length: int = 150) -> \
#     # List[str]:
#     #     # Temporarily disable gradient checkpointing
#     #     self.model.gradient_checkpointing_disable()
#     #     # Project input embeddings to model's hidden size
#     #     projected_embeddings = self.input_projection(input_embeddings)
#     #     projected_embeddings = projected_embeddings.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
#     #
#     #     # Get token embeddings for the prompt_input_ids
#     #     token_embeddings = self.model.get_input_embeddings()(prompt_input_ids)
#     #     # Concatenate projected image embeddings with token embeddings
#     #     inputs_embeds = torch.cat([projected_embeddings, token_embeddings], dim=1)
#     #     # Create attention mask
#     #     batch_size, seq_length, _ = inputs_embeds.size()
#     #     attention_mask = torch.ones((batch_size, seq_length), device=input_embeddings.device, dtype=torch.long)
#     #
#     #     # Generate text
#     #     outputs = self.model.generate(
#     #         inputs_embeds=inputs_embeds,
#     #         attention_mask=attention_mask,
#     #         max_length=seq_length + max_length,
#     #         num_return_sequences=1,
#     #         do_sample=True,
#     #         top_k=50,
#     #         top_p=0.85,
#     #         temperature=0.7,
#     #         pad_token_id=self.tokenizer.pad_token_id,
#     #         eos_token_id=self.tokenizer.eos_token_id,
#     #         use_cache=True,
#     #     )
#     #     # Re-enable gradient checkpointing
#     #     self.model.gradient_checkpointing_enable()
#     #
#     #     # Extract the generated text after the prompt
#     #     generated_texts = []
#     #     for output in outputs:
#     #         generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
#     #         # Remove the prompt from the generated text
#     #         prompt_length = prompt_input_ids.size(1)
#     #         generated_text = generated_text[prompt_length:]
#     #         generated_texts.append(generated_text.strip())
#     #     return generated_texts
#
#
#     def generate_report(self, input_embeddings: torch.Tensor, prompt_input_ids: torch.Tensor, max_length: int = 150) -> \
#     List[str]:
#         """Generate medical reports with improved generation parameters"""
#         self.model.gradient_checkpointing_disable()
#
#         # Project input embeddings
#         projected_embeddings = self.input_projection(input_embeddings)
#         projected_embeddings = projected_embeddings.unsqueeze(1)
#
#         # Get prompt embeddings
#         token_embeddings = self.model.get_input_embeddings()(prompt_input_ids)
#
#         # Concatenate all embeddings
#         inputs_embeds = torch.cat([projected_embeddings, token_embeddings], dim=1)
#
#         # Create attention mask
#         attention_mask = torch.ones((inputs_embeds.size(0), inputs_embeds.size(1)),
#                                     device=input_embeddings.device, dtype=torch.long)
#
#         try:
#             # Generate text with better parameters
#             outputs = self.model.generate(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 max_length=max_length,
#                 min_length=10,  # Ensure some minimum output
#                 num_return_sequences=1,
#                 do_sample=True,
#                 top_k=50,
#                 top_p=0.80,
#                 temperature=1.0,
#                 no_repeat_ngram_size=3,  # Prevent repetition
#                 length_penalty=1.0,
#                 repetition_penalty=1.2,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 bos_token_id=self.tokenizer.bos_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#             )
#
#             # Post-process generated text
#             generated_texts = []
#             for output in outputs:
#                 text = self.tokenizer.decode(output, skip_special_tokens=True)
#                 # Remove prompt from generated text
#                 prompt_text = self.tokenizer.decode(prompt_input_ids[0], skip_special_tokens=True)
#                 if prompt_text in text:
#                     text = text[len(prompt_text):].strip()
#                 if not text:  # If empty after processing, add placeholder
#                     text = "No findings to report."
#                 generated_texts.append(text)
#
#             return generated_texts
#
#         except Exception as e:
#             # logging.error(f"Error in text generation: {e}")
#             return ["Error in report generation"] * input_embeddings.size(0)
#
#         finally:
#             self.model.gradient_checkpointing_enable()


# concat
# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import get_peft_model, LoraConfig, TaskType
# from typing import List
#
# class MedicalReportGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Use BioGPT as the base model
#         self.base_model_name = 'microsoft/biogpt'
#         self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
#         self.model.gradient_checkpointing_enable()
#
#         # PEFT configuration
#         peft_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             inference_mode=False,
#             r=16,
#             lora_alpha=32,
#             lora_dropout=0.1,
#             target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
#         )
#         self.model = get_peft_model(self.model, peft_config)
#
#         # We don't need the input projection layer anymore since we're using concatenated embeddings
#         # directly from the alignment model
#
#         # Ensure special tokens are set
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         if self.tokenizer.bos_token_id is None:
#             self.tokenizer.bos_token = self.tokenizer.eos_token
#         if self.tokenizer.eos_token_id is None:
#             self.tokenizer.eos_token = '</s>'
#             self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
#         self.model.config.pad_token_id = self.tokenizer.pad_token_id
#         self.model.config.bos_token_id = self.tokenizer.bos_token_id
#         self.model.config.eos_token_id = self.tokenizer.eos_token_id
#
#     def forward(self, concatenated_embeddings: torch.Tensor, prompt_input_ids: torch.Tensor, target_ids: torch.Tensor = None):
#         # The concatenated_embeddings already contain the image embeddings, separator, and structural findings
#         # Get token embeddings for the additional prompt text
#         token_embeddings = self.model.get_input_embeddings()(prompt_input_ids)
#
#         # Concatenate everything together
#         inputs_embeds = torch.cat([concatenated_embeddings, token_embeddings], dim=1)
#
#         # Create attention mask for the full sequence
#         attention_mask = torch.ones(inputs_embeds.size()[:2], device=concatenated_embeddings.device, dtype=torch.long)
#
#         if target_ids is not None:
#             # Create labels with -100 for all tokens except target_ids
#             padding_length = concatenated_embeddings.size(1) + prompt_input_ids.size(1)
#             padding = torch.full((target_ids.size(0), padding_length), -100, dtype=torch.long, device=target_ids.device)
#             labels = torch.cat([padding, target_ids], dim=1)
#
#             # Forward pass with labels
#             outputs = self.model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 labels=labels,
#                 return_dict=True
#             )
#             return outputs.loss, outputs.logits
#         else:
#             # For generation
#             outputs = self.model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 return_dict=True
#             )
#             return outputs
#
#     def generate_report(self, concatenated_embeddings: torch.Tensor, prompt_input_ids: torch.Tensor, max_length: int = 150) -> List[str]:
#         """Generate medical reports using concatenated embeddings"""
#         self.model.gradient_checkpointing_disable()
#
#         # Get prompt embeddings
#         token_embeddings = self.model.get_input_embeddings()(prompt_input_ids)
#
#         # Concatenate all embeddings
#         inputs_embeds = torch.cat([concatenated_embeddings, token_embeddings], dim=1)
#
#         # Create attention mask
#         attention_mask = torch.ones((inputs_embeds.size(0), inputs_embeds.size(1)),
#                                     device=concatenated_embeddings.device, dtype=torch.long)
#
#         try:
#             # Generate text
#             outputs = self.model.generate(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 max_length=max_length,
#                 min_length=10,
#                 num_return_sequences=1,
#                 do_sample=True,
#                 top_k=50,
#                 top_p=0.95,
#                 temperature=0.7,
#                 no_repeat_ngram_size=3,
#                 length_penalty=1.0,
#                 repetition_penalty=1.2,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 bos_token_id=self.tokenizer.bos_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#             )
#
#             # Post-process generated text
#             generated_texts = []
#             for output in outputs:
#                 text = self.tokenizer.decode(output, skip_special_tokens=True)
#                 # Remove prompt from generated text
#                 prompt_text = self.tokenizer.decode(prompt_input_ids[0], skip_special_tokens=True)
#                 if prompt_text in text:
#                     text = text[len(prompt_text):].strip()
#                 if not text:
#                     text = "No findings to report."
#                 generated_texts.append(text)
#
#             return generated_texts
#
#         except Exception as e:
#             return ["Error in report generation"] * inputs_embeds.size(0)
#
#         finally:
#             self.model.gradient_checkpointing_enable()

# report_generator.py
# report_generator.py - working concat
# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import get_peft_model, LoraConfig, TaskType
# from typing import List
#
# class MedicalReportGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Use BioGPT as the base model
#         self.base_model_name = 'microsoft/biogpt'
#         self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
#         self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing during training
#
#         # PEFT configuration with target_modules specified
#         peft_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             inference_mode=False,
#             r=16,
#             lora_alpha=32,
#             lora_dropout=0.1,
#             target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
#         )
#         self.model = get_peft_model(self.model, peft_config)
#
#         # Ensure special tokens are set
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         if self.tokenizer.bos_token_id is None:
#             self.tokenizer.bos_token = self.tokenizer.eos_token
#         if self.tokenizer.eos_token_id is None:
#             self.tokenizer.eos_token = '</s>'
#             self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
#         self.model.config.pad_token_id = self.tokenizer.pad_token_id
#         self.model.config.bos_token_id = self.tokenizer.bos_token_id
#         self.model.config.eos_token_id = self.tokenizer.eos_token_id
#
#     def forward(self, combined_embeddings: torch.Tensor, target_ids: torch.Tensor = None):
#         device = combined_embeddings.device
#
#         if target_ids is not None:
#             # Get embeddings for target_ids
#             target_embeddings = self.model.get_input_embeddings()(target_ids)
#
#             # Concatenate 'combined_embeddings' and 'target_embeddings' to form 'inputs_embeds'
#             inputs_embeds = torch.cat([combined_embeddings, target_embeddings], dim=1)
#
#             # Update attention mask
#             attention_mask = torch.ones(inputs_embeds.size()[:2], device=device, dtype=torch.long)
#
#             # Create labels with -100 for image and prompt tokens
#             padding_length = combined_embeddings.size(1)
#             padding = torch.full((target_ids.size(0), padding_length), -100, dtype=torch.long, device=device)
#             labels = torch.cat([padding, target_ids], dim=1)
#
#             # Forward pass with labels
#             outputs = self.model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 labels=labels,
#                 use_cache=False,  # Disable caching when using gradient checkpointing
#                 return_dict=True
#             )
#             return outputs.loss, outputs.logits
#         else:
#             # For generation
#             attention_mask = torch.ones(combined_embeddings.size()[:2], device=device, dtype=torch.long)
#             outputs = self.model.generate(
#                 inputs_embeds=combined_embeddings,
#                 attention_mask=attention_mask,
#                 max_length=combined_embeddings.size(1) + 150,  # Adjust max_length as needed
#                 min_length=10,
#                 num_return_sequences=1,
#                 do_sample=True,
#                 top_k=50,
#                 top_p=0.80,
#                 temperature=1.0,
#                 no_repeat_ngram_size=3,
#                 length_penalty=1.0,
#                 repetition_penalty=1.2,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#                 use_cache=True,  # Enable caching for inference
#             )
#
#             generated_texts = []
#             for output in outputs:
#                 text = self.tokenizer.decode(output, skip_special_tokens=True)
#                 # Remove the input prompt from the generated text
#                 generated_text = text  # Adjust this if needed
#                 generated_texts.append(generated_text.strip())
#
#             return generated_texts
#
#     def generate_report(self, combined_embeddings: torch.Tensor, max_length: int = 150) -> List[str]:
#         """Generate medical reports"""
#         self.model.gradient_checkpointing_disable()
#
#         device = combined_embeddings.device
#         attention_mask = torch.ones(combined_embeddings.size()[:2], device=device, dtype=torch.long)
#
#         try:
#             # Generate text
#             outputs = self.model.generate(
#                 inputs_embeds=combined_embeddings,
#                 attention_mask=attention_mask,
#                 max_length=combined_embeddings.size(1) + max_length,  # adjust max_length accordingly
#                 min_length=10,
#                 num_return_sequences=1,
#                 do_sample=True,
#                 top_k=50,
#                 top_p=0.9,
#                 temperature=0.8,
#                 no_repeat_ngram_size=3,
#                 length_penalty=1.0,
#                 repetition_penalty=1.0,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#                 use_cache=True,
#             )
#
#             # Decode generated texts
#             generated_texts = []
#             for output in outputs:
#                 text = self.tokenizer.decode(output, skip_special_tokens=True)
#                 generated_texts.append(text.strip())
#
#             return generated_texts
#
#         except Exception as e:
#             return ["Error in report generation"] * combined_embeddings.size(0)
#
#         finally:
#             self.model.gradient_checkpointing_enable()

# corrected alignment
# report_generator.py

# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import get_peft_model, LoraConfig, TaskType
# from typing import List
#
# class MedicalReportGenerator(nn.Module):
#     def __init__(self, image_embedding_dim=512):
#         super().__init__()
#         # Use BioGPT as the base model
#         self.base_model_name = 'microsoft/biogpt'
#         self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
#         self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing during training
#
#         # PEFT configuration with target_modules specified
#         peft_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             inference_mode=False,
#             r=16,
#             lora_alpha=32,
#             lora_dropout=0.1,
#             target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
#         )
#         self.model = get_peft_model(self.model, peft_config)
#
#         # Projection layer to map image embeddings to model's embedding size
#         self.image_projection = nn.Linear(image_embedding_dim, self.model.config.hidden_size)
#
#         # Token embeddings for separator token
#         if 'sep_token' not in self.tokenizer.special_tokens_map:
#             self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})
#             self.model.resize_token_embeddings(len(self.tokenizer))
#
#         self.sep_token_id = self.tokenizer.sep_token_id
#
#         # Ensure special tokens are set
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         if self.tokenizer.bos_token_id is None:
#             self.tokenizer.bos_token = self.tokenizer.eos_token
#         if self.tokenizer.eos_token_id is None:
#             self.tokenizer.eos_token = '</s>'
#             self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
#         self.model.config.pad_token_id = self.tokenizer.pad_token_id
#         self.model.config.bos_token_id = self.tokenizer.bos_token_id
#         self.model.config.eos_token_id = self.tokenizer.eos_token_id
#
#     def forward(self, image_embeddings: torch.Tensor, prompt_input_ids: torch.Tensor, target_ids: torch.Tensor = None):
#         # Project image embeddings and add sequence dimension
#         projected_embeddings = self.image_projection(image_embeddings).unsqueeze(1)  # (batch_size, 1, hidden_size)
#
#         # Get separator embedding
#         sep_embedding = self.model.get_input_embeddings()(torch.tensor([self.sep_token_id], device=image_embeddings.device))
#         sep_embedding = sep_embedding.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)  # (batch_size, 1, hidden_size)
#
#         # Combine image and separator embeddings
#         image_and_sep_embeddings = torch.cat([projected_embeddings, sep_embedding], dim=1)  # (batch_size, 2, hidden_size)
#
#         if target_ids is not None:
#             # Concatenate prompt and target input IDs
#             full_input_ids = torch.cat([prompt_input_ids, target_ids], dim=1)  # (batch_size, seq_len_prompt + seq_len_target)
#
#             # Get embeddings for the prompt and target
#             token_embeddings = self.model.get_input_embeddings()(full_input_ids)  # (batch_size, seq_len_prompt + seq_len_target, hidden_size)
#
#             # Concatenate all embeddings
#             inputs_embeds = torch.cat([image_and_sep_embeddings, token_embeddings], dim=1)  # (batch_size, total_seq_len, hidden_size)
#
#             # Create attention mask
#             attention_mask = torch.ones(inputs_embeds.size()[:2], device=inputs_embeds.device, dtype=torch.long)
#
#             # Create labels with -100 for image, separator, and prompt tokens
#             labels = torch.full((inputs_embeds.size(0), inputs_embeds.size(1)), -100, dtype=torch.long, device=inputs_embeds.device)
#             labels[:, image_and_sep_embeddings.size(1) + prompt_input_ids.size(1):] = target_ids  # Only compute loss for target tokens
#
#             # Forward pass with labels
#             outputs = self.model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 labels=labels,
#                 return_dict=True
#             )
#
#             return outputs.loss, outputs.logits
#         else:
#             # For generation
#             token_embeddings = self.model.get_input_embeddings()(prompt_input_ids)  # (batch_size, seq_len_prompt, hidden_size)
#             inputs_embeds = torch.cat([image_and_sep_embeddings, token_embeddings], dim=1)
#
#             attention_mask = torch.ones(inputs_embeds.size()[:2], device=inputs_embeds.device, dtype=torch.long)
#
#             outputs = self.model.generate(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 max_length=inputs_embeds.size(1) + 150,
#                 min_length=inputs_embeds.size(1) + 10,
#                 num_return_sequences=1,
#                 do_sample=True,
#                 top_k=50,
#                 top_p=0.80,
#                 temperature=1.0,
#                 no_repeat_ngram_size=3,
#                 length_penalty=1.0,
#                 repetition_penalty=1.2,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 bos_token_id=self.tokenizer.bos_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#             )
#
#             # Decode generated tokens
#             generated_texts = []
#             for output in outputs:
#                 text = self.tokenizer.decode(output, skip_special_tokens=True)
#                 # Optionally, remove the prompt from the generated text
#                 generated_texts.append(text)
#
#             return generated_texts

# BioGPT full data updated metrics
# report_generator.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from typing import List

class MedicalReportGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Use BioGPT as the base model
        self.base_model_name = 'microsoft/biogpt'
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing during training

        # PEFT configuration with target_modules specified
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)

        # Projection layer to map image embeddings to model's embedding size
        self.input_projection = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

        # Ensure special tokens are set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token = '</s>'
            self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

    def forward(self, input_embeddings: torch.Tensor, target_ids: torch.Tensor = None):
        # Project input embeddings to model's hidden size
        projected_embeddings = self.input_projection(input_embeddings)
        projected_embeddings = projected_embeddings.unsqueeze(1)  # Add sequence dimension

        if target_ids is not None:
            # Get token embeddings for the target sequence
            token_embeddings = self.model.get_input_embeddings()(target_ids)
            # Concatenate projected image embeddings with token embeddings
            inputs_embeds = torch.cat([projected_embeddings, token_embeddings], dim=1)
            # Adjust attention mask
            attention_mask = torch.ones(inputs_embeds.size()[:2], device=input_embeddings.device, dtype=torch.long)
            # Pad labels with -100 at the beginning to match input length
            padding = torch.full((target_ids.size(0), 1), -100, dtype=torch.long, device=target_ids.device)
            labels = torch.cat([padding, target_ids], dim=1)
            # Forward pass with labels
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            return outputs.loss, outputs.logits
        else:
            raise ValueError("Target IDs must be provided during training.")

    def generate_report(self, input_embeddings: torch.Tensor, max_length: int = 150) -> List[str]:
        # Temporarily disable gradient checkpointing
        self.model.gradient_checkpointing_disable()
        # Project input embeddings to model's hidden size
        projected_embeddings = self.input_projection(input_embeddings)
        projected_embeddings = projected_embeddings.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)

        # Get BOS token id
        bos_token_id = self.tokenizer.bos_token_id
        if bos_token_id is None:
            raise ValueError("bos_token_id is not set in the tokenizer.")

        # Get embedding of BOS token
        bos_embedding = self.model.get_input_embeddings()(torch.tensor([[bos_token_id]]).to(input_embeddings.device))
        # Shape: (1, 1, hidden_size)

        # Repeat bos_embedding for batch size
        bos_embedding = bos_embedding.expand(input_embeddings.size(0), -1, -1)  # Shape: (batch_size, 1, hidden_size)

        # Concatenate bos_embedding and projected_embeddings
        inputs_embeds = torch.cat([bos_embedding, projected_embeddings], dim=1)  # Shape: (batch_size, 2, hidden_size)

        # Create attention mask
        batch_size = inputs_embeds.size(0)
        attention_mask = torch.ones((batch_size, inputs_embeds.size(1)), device=inputs_embeds.device, dtype=torch.long)

        # Generate text
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=10,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.85,
            temperature=0.8,
            length_penalty=1.0,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        # Re-enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts

# Concat-Trail2
# report_generator.py

# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import get_peft_model, LoraConfig, TaskType
# from typing import List
#
# class MedicalReportGenerator(nn.Module):
#     def __init__(self, image_embedding_dim=512):
#         super().__init__()
#         # Use BioGPT as the base model
#         self.base_model_name = 'microsoft/biogpt'
#         self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
#         self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing during training
#
#         # PEFT configuration with target_modules specified
#         peft_config = LoraConfig(
#             task_type=TaskType.CAUSAL_LM,
#             inference_mode=False,
#             r=16,
#             lora_alpha=32,
#             lora_dropout=0.1,
#             target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
#         )
#         self.model = get_peft_model(self.model, peft_config)
#
#         # Projection layer to map image embeddings to model's embedding size
#         self.image_projection = nn.Linear(image_embedding_dim, self.model.config.hidden_size)
#
#         # Token embeddings for separator token
#         if 'sep_token' not in self.tokenizer.special_tokens_map:
#             self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})
#             self.model.resize_token_embeddings(len(self.tokenizer))
#
#         self.sep_token_id = self.tokenizer.sep_token_id
#
#         # Ensure special tokens are set
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#         if self.tokenizer.bos_token_id is None:
#             self.tokenizer.bos_token = self.tokenizer.eos_token
#         if self.tokenizer.eos_token_id is None:
#             self.tokenizer.eos_token = '</s>'
#             self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
#         self.model.config.pad_token_id = self.tokenizer.pad_token_id
#         self.model.config.bos_token_id = self.tokenizer.bos_token_id
#         self.model.config.eos_token_id = self.tokenizer.eos_token_id
#
#     def forward(self, image_embeddings: torch.Tensor, prompt_input_ids: torch.Tensor, target_ids: torch.Tensor = None):
#         # Project image embeddings and add sequence dimension
#         projected_embeddings = self.image_projection(image_embeddings).unsqueeze(1)  # (batch_size, 1, hidden_size)
#
#         # Get separator embedding
#         sep_embedding = self.model.get_input_embeddings()(torch.tensor([self.sep_token_id], device=image_embeddings.device))
#         sep_embedding = sep_embedding.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)  # (batch_size, 1, hidden_size)
#
#         # Combine image and separator embeddings
#         image_and_sep_embeddings = torch.cat([projected_embeddings, sep_embedding], dim=1)  # (batch_size, 2, hidden_size)
#
#         if target_ids is not None:
#             # Concatenate prompt and target input IDs
#             full_input_ids = torch.cat([prompt_input_ids, target_ids], dim=1)  # (batch_size, seq_len_prompt + seq_len_target)
#
#             # Get embeddings for the prompt and target
#             token_embeddings = self.model.get_input_embeddings()(full_input_ids)  # (batch_size, seq_len_prompt + seq_len_target, hidden_size)
#
#             # Concatenate all embeddings
#             inputs_embeds = torch.cat([image_and_sep_embeddings, token_embeddings], dim=1)  # (batch_size, total_seq_len, hidden_size)
#
#             # Create attention mask
#             attention_mask = torch.ones(inputs_embeds.size()[:2], device=inputs_embeds.device, dtype=torch.long)
#
#             # Create labels with -100 for image, separator, and prompt tokens
#             labels = torch.full((inputs_embeds.size(0), inputs_embeds.size(1)), -100, dtype=torch.long, device=inputs_embeds.device)
#             labels[:, image_and_sep_embeddings.size(1) + prompt_input_ids.size(1):] = target_ids  # Only compute loss for target tokens
#
#             # Forward pass with labels
#             outputs = self.model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 labels=labels,
#                 return_dict=True
#             )
#
#             return outputs.loss, outputs.logits
#         else:
#             # For generation
#             token_embeddings = self.model.get_input_embeddings()(prompt_input_ids)  # (batch_size, seq_len_prompt, hidden_size)
#             inputs_embeds = torch.cat([image_and_sep_embeddings, token_embeddings], dim=1)
#
#             attention_mask = torch.ones(inputs_embeds.size()[:2], device=inputs_embeds.device, dtype=torch.long)
#
#             outputs = self.model.generate(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 max_length=inputs_embeds.size(1) + 150,
#                 min_length=inputs_embeds.size(1) + 10,
#                 num_return_sequences=1,
#                 do_sample=True,
#                 top_k=50,
#                 top_p=0.85,
#                 temperature=0.8,
#                 no_repeat_ngram_size=3,
#                 length_penalty=1.0,
#                 repetition_penalty=1.2,
#                 pad_token_id=self.tokenizer.pad_token_id,
#                 bos_token_id=self.tokenizer.bos_token_id,
#                 eos_token_id=self.tokenizer.eos_token_id,
#             )
#
#             # Decode generated tokens
#             generated_texts = []
#             for output in outputs:
#                 text = self.tokenizer.decode(output, skip_special_tokens=True)
#                 # Optionally, remove the prompt from the generated text
#                 generated_texts.append(text)
#
#             return generated_texts
