import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm

class ImageTextAlignmentModel(nn.Module):
  def __init__(self, image_embedding_dim: int = 512):  # Changed from 2048 to 512
      super().__init__()

      # Load Bio-BERT with trust_remote_code=True
      self.text_encoder = AutoModel.from_pretrained(
          'microsoft/BiomedVLP-CXR-BERT-specialized',
          trust_remote_code=True
      )
      self.tokenizer = AutoTokenizer.from_pretrained(
          'microsoft/BiomedVLP-CXR-BERT-specialized',
          trust_remote_code=True
      )

      # Projection layers
      self.image_projection = nn.Linear(image_embedding_dim, 768)  # Project from 512 to 768
      self.text_projection = nn.Linear(768, 768)

  def forward(self, image_embeddings: torch.Tensor, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
      # Project image embeddings
      projected_image = self.image_projection(image_embeddings)

      # Encode text
      text_encoding = self.tokenizer(
          text, 
          padding=True, 
          truncation=True, 
          return_tensors="pt",
          max_length=512
      )
      text_encoding = {k: v.to(image_embeddings.device) for k, v in text_encoding.items()}

      text_features = self.text_encoder(**text_encoding).last_hidden_state[:, 0, :]  # Use [CLS] token
      projected_text = self.text_projection(text_features)

      return projected_image, projected_text