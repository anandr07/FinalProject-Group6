# report_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Tuple, Dict, Optional, Union
import logging

class MedicalReportGenerator(nn.Module):
  def __init__(self):
      super().__init__()
      # Use GPT-2 as the base model
      self.model = GPT2LMHeadModel.from_pretrained('gpt2')
      self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

      # Add padding token
      self.tokenizer.pad_token = self.tokenizer.eos_token
      self.model.config.pad_token_id = self.model.config.eos_token_id

      # Add projection layer from BERT hidden size (768) to GPT2 hidden size (768)
      self.input_projection = nn.Linear(768, self.model.config.n_embd)

  def forward(self, input_embeddings: torch.Tensor, target_ids: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
      batch_size = input_embeddings.size(0)

      # Project input embeddings to GPT2 hidden size
      projected_embeddings = self.input_projection(input_embeddings)
      projected_embeddings = projected_embeddings.unsqueeze(1)  # Add sequence dimension

      if target_ids is not None:
          # Create position IDs and attention mask for the full sequence
          seq_length = target_ids.size(1)
          position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_embeddings.device)
          position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
          attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=input_embeddings.device)

          # Get token embeddings for the target sequence
          token_embeddings = self.model.transformer.wte(target_ids)

          # Concatenate projected image embeddings with token embeddings
          inputs_embeds = torch.cat([projected_embeddings, token_embeddings[:, :-1]], dim=1)

          # Adjust attention mask and position ids for the combined sequence
          full_attention_mask = torch.ones((batch_size, inputs_embeds.size(1)),
                                        dtype=torch.long,
                                        device=input_embeddings.device)
          full_position_ids = torch.arange(0, inputs_embeds.size(1),
                                        dtype=torch.long,
                                        device=input_embeddings.device).expand(batch_size, -1)

          # Forward pass with labels
          outputs = self.model(
              inputs_embeds=inputs_embeds,
              attention_mask=full_attention_mask,
              position_ids=full_position_ids,
              labels=target_ids,
              return_dict=True
          )
          return outputs.loss, outputs.logits
      else:
          # For generation
          outputs = self.model(
              inputs_embeds=projected_embeddings,
              return_dict=True
          )
          return outputs.logits

  def generate_report(self, input_embeddings: torch.Tensor, max_length: int = 150) -> List[str]:
      # Project input embeddings
      projected_embeddings = self.input_projection(input_embeddings)
      projected_embeddings = projected_embeddings.unsqueeze(1)  # Add sequence dimension

      # Create attention mask for the input
      batch_size = input_embeddings.size(0)
      attention_mask = torch.ones(batch_size, 1, dtype=torch.long, device=input_embeddings.device)

      # Generate text
      outputs = self.model.generate(
          inputs_embeds=projected_embeddings,
          max_length=max_length,
          num_return_sequences=1,
          no_repeat_ngram_size=3,
          do_sample=True,
          top_k=50,
          top_p=0.95,
          temperature=0.7,
          pad_token_id=self.tokenizer.pad_token_id,
          eos_token_id=self.tokenizer.eos_token_id,
          attention_mask=attention_mask
      )

      return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)