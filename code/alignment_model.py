# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import AutoModel, AutoTokenizer, AutoConfig
# from typing import List, Tuple, Dict, Optional
# import numpy as np
# from tqdm import tqdm
#
# class ImageTextAlignmentModel(nn.Module):
#   def __init__(self, image_embedding_dim: int = 512):  # Changed from 2048 to 512
#       super().__init__()
#
#       # Load Bio-BERT with trust_remote_code=True
#       self.text_encoder = AutoModel.from_pretrained(
#           'microsoft/BiomedVLP-CXR-BERT-specialized',
#           trust_remote_code=True
#       )
#       self.tokenizer = AutoTokenizer.from_pretrained(
#           'microsoft/BiomedVLP-CXR-BERT-specialized',
#           trust_remote_code=True
#       )
#
#       # Projection layers
#       self.image_projection = nn.Linear(image_embedding_dim, 768)  # Project from 512 to 768
#       self.text_projection = nn.Linear(768, 768)
#
#   def forward(self, image_embeddings: torch.Tensor, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
#       # Project image embeddings
#       projected_image = self.image_projection(image_embeddings)
#
#       # Encode text
#       text_encoding = self.tokenizer(
#           text,
#           padding=True,
#           truncation=True,
#           return_tensors="pt",
#           max_length=512
#       )
#       text_encoding = {k: v.to(image_embeddings.device) for k, v in text_encoding.items()}
#
#       text_features = self.text_encoder(**text_encoding).last_hidden_state[:, 0, :]  # Use [CLS] token
#       projected_text = self.text_projection(text_features)
#
#       return projected_image, projected_text
# biogpt
# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer
# from typing import List, Tuple
#
# class ImageTextAlignmentModel(nn.Module):
#     def __init__(self, image_embedding_dim: int = 512):  # Changed from 2048 to 512
#         super().__init__()
#
#         # Load BioGPT and its tokenizer
#         self.text_encoder = AutoModel.from_pretrained('microsoft/biogpt')
#         self.tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
#
#         # Projection layers
#         self.image_projection = nn.Linear(image_embedding_dim, self.text_encoder.config.hidden_size)  # Align with BioGPT
#         self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size)
#
#     def forward(self, image_embeddings: torch.Tensor, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Project image embeddings to match the text embedding size
#         projected_image = self.image_projection(image_embeddings)
#
#         # Tokenize text inputs
#         text_encoding = self.tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512
#         )
#         text_encoding = {k: v.to(image_embeddings.device) for k, v in text_encoding.items()}
#
#         # Pass text through BioGPT encoder
#         text_features = self.text_encoder(**text_encoding).last_hidden_state[:, 0, :]  # Use [CLS] token
#         projected_text = self.text_projection(text_features)
#
#         return projected_image, projected_text

#Trail: CITE
# alignment_model.py
#
# import torch
# import torch.nn as nn
# from transformers import AutoImageProcessor, AutoModel
#
# class ImageTextEmbeddingModel(nn.Module):
#   def __init__(self):
#       super().__init__()
#
#       # Load the pre-trained microsoft/rad-dino model
#       self.model_name = 'microsoft/rad-dino'
#       self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
#       self.rad_model = AutoModel.from_pretrained(self.model_name)
#
#       # Get the embedding dimension (hidden size)
#       self.embedding_dim = self.rad_model.config.hidden_size
#
#       # Freeze all parameters
#       for param in self.rad_model.parameters():
#           param.requires_grad = False
#
#       # Optionally, unfreeze the final layer or specific layers
#       # For illustration, let's unfreeze the last transformer block
#       for param in self.rad_model.encoder.layer[-1].parameters():
#           param.requires_grad = True
#
#   def forward(self, images: torch.Tensor) -> torch.Tensor:
#       # Prepare inputs
#       outputs = self.rad_model(pixel_values=images)
#       # Get the pooled output
#       image_embeddings = outputs.pooler_output  # Shape: (batch_size, hidden_size)
#       return image_embeddings

# Prompt BioGPT:
# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer
# from typing import List, Tuple
#
# class ImageTextAlignmentModel(nn.Module):
#     def __init__(self, image_embedding_dim: int = 512):  # Adjusted to match BioViLT output dimension
#         super().__init__()
#
#         # Load BioGPT and its tokenizer
#         self.text_encoder = AutoModel.from_pretrained('microsoft/biogpt')
#         self.tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
#
#         # Projection layers
#         self.image_projection = nn.Linear(image_embedding_dim, self.text_encoder.config.hidden_size)  # Align with BioGPT
#         self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size)
#
#     def forward(self, image_embeddings: torch.Tensor, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Project image embeddings to match the text embedding size
#         projected_image = self.image_projection(image_embeddings)
#
#         # Tokenize text inputs
#         text_encoding = self.tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512
#         )
#         text_encoding = {k: v.to(image_embeddings.device) for k, v in text_encoding.items()}
#
#         # Pass text through BioGPT encoder
#         text_features = self.text_encoder(**text_encoding).last_hidden_state[:, 0, :]  # Use first token embedding
#         projected_text = self.text_projection(text_features)
#
#         return projected_image, projected_text

# Trail - c
# alignment_model.py
# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer
# from typing import List, Tuple, Optional
# import torch.nn.functional as F
# import math
#
#
# class CrossAttention(nn.Module):
#     def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)
#         self.out_proj = nn.Linear(dim, dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
#                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         batch_size = query.size(0)
#
#         # Linear projections and reshape
#         q = self.q_proj(query).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         k = self.k_proj(key).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         v = self.v_proj(value).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#
#         # Attention
#         scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))
#         attn_weights = F.softmax(scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)
#
#         # Combine values
#         attn_output = torch.matmul(attn_weights, v)
#         attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
#
#         return self.out_proj(attn_output)
#
#
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, max_len: int = 5000):
#         super().__init__()
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         return x + self.pe[:x.size(0)]
#
#
# class ImageTextAlignmentModel(nn.Module):
#     def __init__(self, image_embedding_dim: int = 512, text_embedding_dim: Optional[int] = None):
#         super().__init__()
#
#         # Initialize BioGPT encoder and tokenizer
#         self.text_encoder = AutoModel.from_pretrained('microsoft/biogpt')
#         self.tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
#
#         if text_embedding_dim is None:
#             text_embedding_dim = self.text_encoder.config.hidden_size
#
#         # Projection networks with layer normalization
#         self.image_projection = nn.Sequential(
#             nn.Linear(image_embedding_dim, text_embedding_dim * 2),
#             nn.LayerNorm(text_embedding_dim * 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(text_embedding_dim * 2, text_embedding_dim),
#             nn.LayerNorm(text_embedding_dim)
#         )
#
#         self.text_projection = nn.Sequential(
#             nn.Linear(text_embedding_dim, text_embedding_dim * 2),
#             nn.LayerNorm(text_embedding_dim * 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(text_embedding_dim * 2, text_embedding_dim),
#             nn.LayerNorm(text_embedding_dim)
#         )
#
#         # Cross-attention layers
#         self.image_to_text_attention = CrossAttention(text_embedding_dim)
#         self.text_to_image_attention = CrossAttention(text_embedding_dim)
#
#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(text_embedding_dim)
#
#         # Final fusion layers
#         self.fusion_layer = nn.Sequential(
#             nn.Linear(text_embedding_dim * 3, text_embedding_dim * 2),
#             nn.LayerNorm(text_embedding_dim * 2),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(text_embedding_dim * 2, text_embedding_dim)
#         )
#
#         # Output normalization
#         self.output_norm = nn.LayerNorm(text_embedding_dim)
#
#         # Initialize weights
#         self._init_weights()
#
#     def _init_weights(self):
#         """Initialize weights with Xavier uniform distribution"""
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#
#     def encode_text(self, text: List[str], device: torch.device) -> torch.Tensor:
#         """Encode text using BioGPT"""
#         # Tokenize and encode text
#         text_encoding = self.tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512
#         ).to(device)
#
#         # Get text features
#         with torch.no_grad():
#             text_outputs = self.text_encoder(**text_encoding)
#             text_features = text_outputs.last_hidden_state
#
#         # Add positional encoding
#         text_features = text_features.transpose(0, 1)
#         text_features = self.pos_encoder(text_features)
#         text_features = text_features.transpose(0, 1)
#
#         return text_features
#
#     def forward(self, image_embeddings: torch.Tensor, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Get device
#         device = image_embeddings.device
#
#         # Encode text
#         text_features = self.encode_text(text, device)
#
#         # Project features
#         projected_image = self.image_projection(image_embeddings)
#         projected_text = self.text_projection(text_features)
#
#         # Cross attention
#         image_attended = self.text_to_image_attention(
#             query=projected_image.unsqueeze(1),
#             key=projected_text,
#             value=projected_text
#         ).squeeze(1)
#
#         text_attended = self.image_to_text_attention(
#             query=projected_text,
#             key=projected_image.unsqueeze(1),
#             value=projected_image.unsqueeze(1)
#         )
#
#         # Global text representation
#         text_pooled = torch.mean(text_attended, dim=1)
#
#         # Concatenate and fuse features
#         fused_features = torch.cat([
#             projected_image,
#             image_attended,
#             text_pooled
#         ], dim=-1)
#
#         # Final projection
#         output = self.fusion_layer(fused_features)
#         output = self.output_norm(output)
#
#         return output, text_pooled
#
#     def get_attention_weights(self, image_embeddings: torch.Tensor, text: List[str]) -> Tuple[
#         torch.Tensor, torch.Tensor]:
#         """Get attention weights for visualization (useful for debugging)"""
#         device = image_embeddings.device
#
#         # Encode and project features
#         text_features = self.encode_text(text, device)
#         projected_image = self.image_projection(image_embeddings)
#         projected_text = self.text_projection(text_features)
#
#         # Calculate attention weights
#         with torch.no_grad():
#             img_to_text_attn = self.image_to_text_attention(
#                 projected_image.unsqueeze(1),
#                 projected_text,
#                 projected_text,
#             )
#             text_to_img_attn = self.text_to_image_attention(
#                 projected_text,
#                 projected_image.unsqueeze(1),
#                 projected_image.unsqueeze(1),
#             )
#
#         return img_to_text_attn, text_to_img_attn

# Concat
# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer
# from typing import List, Tuple
#
#
# class ImageTextAlignmentModel(nn.Module):
#     def __init__(self, image_embedding_dim: int = 512, text_embedding_dim: int = None):
#         super().__init__()
#
#         # Initialize BioGPT encoder and tokenizer
#         self.text_encoder = AutoModel.from_pretrained('microsoft/biogpt')
#         self.tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
#
#         if text_embedding_dim is None:
#             text_embedding_dim = self.text_encoder.config.hidden_size
#
#         # Projection networks with layer normalization
#         self.image_projection = nn.Sequential(
#             nn.Linear(image_embedding_dim, text_embedding_dim),
#             nn.LayerNorm(text_embedding_dim),
#             nn.GELU(),
#             nn.Dropout(0.1)
#         )
#
#         self.text_projection = nn.Sequential(
#             nn.Linear(text_embedding_dim, text_embedding_dim),
#             nn.LayerNorm(text_embedding_dim),
#             nn.GELU(),
#             nn.Dropout(0.1)
#         )
#
#         # Output normalization
#         self.output_norm = nn.LayerNorm(text_embedding_dim)
#
#         # Initialize weights
#         self._init_weights()
#
#         # Add separator embedding
#         self.separator_embedding = nn.Parameter(torch.randn(1, 1, text_embedding_dim))
#         nn.init.normal_(self.separator_embedding, std=0.02)
#
#     def _init_weights(self):
#         """Initialize weights with Xavier uniform distribution"""
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#
#     def encode_text(self, text: List[str], device: torch.device) -> torch.Tensor:
#         """Encode text using BioGPT"""
#         # Tokenize and encode text
#         text_encoding = self.tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512
#         ).to(device)
#
#         # Get text features
#         with torch.no_grad():
#             text_outputs = self.text_encoder(**text_encoding)
#             text_features = text_outputs.last_hidden_state
#
#         return text_features
#
#     def forward(self, image_embeddings: torch.Tensor, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Get device
#         device = image_embeddings.device
#
#         # Encode text
#         text_features = self.encode_text(text, device)
#
#         # Project features
#         projected_image = self.image_projection(image_embeddings)
#         projected_text = self.text_projection(text_features)
#
#         # Expand separator embedding to match batch size
#         batch_size = projected_image.size(0)
#         separator = self.separator_embedding.expand(batch_size, -1, -1)
#
#         # Simple concatenation with separator
#         # Shape: (batch_size, sequence_length, hidden_size)
#         concatenated_features = torch.cat([
#             projected_image.unsqueeze(1),  # Add sequence dimension
#             separator,
#             projected_text
#         ], dim=1)
#
#         # Apply final normalization
#         output = self.output_norm(concatenated_features)
#
#         # Get pooled text representation for loss calculation
#         text_pooled = torch.mean(projected_text, dim=1)
#
#         return output, text_pooled

# alignment_model.py - working concat
# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer
# from typing import List, Tuple, Optional
#
#
# class ImageTextAlignmentModel(nn.Module):
#     def __init__(self, image_embedding_dim: int = 512, text_embedding_dim: Optional[int] = None):
#         super().__init__()
#
#         # Initialize BioGPT encoder and tokenizer
#         self.text_encoder = AutoModel.from_pretrained('microsoft/biogpt')
#         self.tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
#
#         if text_embedding_dim is None:
#             text_embedding_dim = self.text_encoder.config.hidden_size
#
#         # Projection network for image embeddings
#         self.image_projection = nn.Sequential(
#             nn.Linear(image_embedding_dim, text_embedding_dim),
#             nn.LayerNorm(text_embedding_dim)
#         )
#
#         # Separator token
#         self.sep_token = '[SEP]'  # You can choose 'prompt' or any other token
#         if self.sep_token not in self.tokenizer.get_vocab():
#             self.tokenizer.add_tokens([self.sep_token])
#             self.text_encoder.resize_token_embeddings(len(self.tokenizer))
#
#         # Initialize weights
#         self._init_weights()
#
#     def _init_weights(self):
#         """Initialize weights with Xavier uniform distribution"""
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#
#     def encode_text(self, text: List[str], device: torch.device) -> torch.Tensor:
#         """Encode text using BioGPT"""
#         # Tokenize and encode text
#         text_encoding = self.tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512
#         ).to(device)
#
#         # Get text embeddings
#         with torch.no_grad():
#             text_outputs = self.text_encoder(**text_encoding)
#             text_embeddings = text_outputs.last_hidden_state
#
#         return text_embeddings
#
#     def forward(self, image_embeddings: torch.Tensor, text: List[str]) -> torch.Tensor:
#         device = image_embeddings.device
#
#         # Encode text embeddings
#         text_embeddings = self.encode_text(text, device)
#
#         # Project image embeddings to the same dimension as text embeddings
#         projected_image_embeddings = self.image_projection(image_embeddings)
#         projected_image_embeddings = projected_image_embeddings.unsqueeze(1)  # Add sequence dimension
#
#         # Get [SEP] token embedding
#         sep_token_id = self.tokenizer.convert_tokens_to_ids(self.sep_token)
#         sep_token_embedding = self.text_encoder.get_input_embeddings()(torch.tensor([sep_token_id]).to(device))
#         sep_token_embedding = sep_token_embedding.unsqueeze(0).repeat(image_embeddings.size(0), 1, 1)
#
#         # Concatenate embeddings: [Image Embedding] + [SEP] + [Text Embeddings]
#         combined_embeddings = torch.cat([projected_image_embeddings, sep_token_embedding, text_embeddings], dim=1)
#
#         return combined_embeddings

# corrected alignment
# alignment_model.py

# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer
# from typing import List, Tuple, Optional
#
# class ImageTextAlignmentModel(nn.Module):
#     def __init__(self, image_embedding_dim: int = 512, text_embedding_dim: Optional[int] = None):
#         super().__init__()
#
#         # Initialize BioGPT encoder and tokenizer
#         self.text_encoder = AutoModel.from_pretrained('microsoft/biogpt')
#         self.tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
#
#         if text_embedding_dim is None:
#             text_embedding_dim = self.text_encoder.config.hidden_size
#
#         # Projection networks with layer normalization
#         self.image_projection = nn.Sequential(
#             nn.Linear(image_embedding_dim, text_embedding_dim),
#             nn.LayerNorm(text_embedding_dim),
#             nn.GELU(),
#         )
#
#         self.text_projection = nn.Sequential(
#             nn.Linear(text_embedding_dim, text_embedding_dim),
#             nn.LayerNorm(text_embedding_dim),
#             nn.GELU(),
#         )
#
#         # Initialize weights
#         self._init_weights()
#
#     def _init_weights(self):
#         """Initialize weights with Xavier uniform distribution"""
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#
#     def encode_text(self, text: List[str], device: torch.device) -> torch.Tensor:
#         """Encode text using BioGPT"""
#         # Tokenize and encode text
#         text_encoding = self.tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512
#         ).to(device)
#
#         # Get text features
#         with torch.no_grad():
#             text_outputs = self.text_encoder(**text_encoding)
#             text_features = text_outputs.last_hidden_state[:, 0, :]  # Take [CLS] token
#
#         return text_features
#
#     def forward(self, image_embeddings: torch.Tensor, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Get device
#         device = image_embeddings.device
#
#         # Encode text
#         text_features = self.encode_text(text, device)
#
#         # Project features
#         projected_image = self.image_projection(image_embeddings)
#         projected_text = self.text_projection(text_features)
#
#         return projected_image, projected_text

# BioGPT full data updated metrics

# alignment_model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple

class ImageTextAlignmentModel(nn.Module):
    def __init__(self, image_embedding_dim: int = 512):  # Changed from 2048 to 512
        super().__init__()

        # Load BioGPT and its tokenizer
        self.text_encoder = AutoModel.from_pretrained('microsoft/biogpt')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')

        # Projection layers
        self.image_projection = nn.Linear(image_embedding_dim, self.text_encoder.config.hidden_size)  # Align with BioGPT
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size)

    def forward(self, image_embeddings: torch.Tensor, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project image embeddings to match the text embedding size
        projected_image = self.image_projection(image_embeddings)

        # Tokenize text inputs
        text_encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        text_encoding = {k: v.to(image_embeddings.device) for k, v in text_encoding.items()}

        # Pass text through BioGPT encoder
        text_features = self.text_encoder(**text_encoding).last_hidden_state[:, 0, :]  # Use [CLS] token
        projected_text = self.text_projection(text_features)

        return projected_image, projected_text

# Concat Trail 2
# alignment_model.py

# import torch
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer
# from typing import List, Tuple, Optional
#
# class ImageTextAlignmentModel(nn.Module):
#     def __init__(self, image_embedding_dim: int = 512, text_embedding_dim: Optional[int] = None):
#         super().__init__()
#
#         # Initialize BioGPT encoder and tokenizer
#         self.text_encoder = AutoModel.from_pretrained('microsoft/biogpt')
#         self.tokenizer = AutoTokenizer.from_pretrained('microsoft/biogpt')
#
#         if text_embedding_dim is None:
#             text_embedding_dim = self.text_encoder.config.hidden_size
#
#         # Projection networks with layer normalization
#         self.image_projection = nn.Sequential(
#             nn.Linear(image_embedding_dim, text_embedding_dim),
#             nn.LayerNorm(text_embedding_dim),
#             nn.GELU(),
#         )
#
#         self.text_projection = nn.Sequential(
#             nn.Linear(text_embedding_dim, text_embedding_dim),
#             nn.LayerNorm(text_embedding_dim),
#             nn.GELU(),
#         )
#
#         # Initialize weights
#         self._init_weights()
#
#     def _init_weights(self):
#         """Initialize weights with Xavier uniform distribution"""
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
#
#     def encode_text(self, text: List[str], device: torch.device) -> torch.Tensor:
#         """Encode text using BioGPT"""
#         # Tokenize and encode text
#         text_encoding = self.tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512
#         ).to(device)
#
#         # Get text features
#         with torch.no_grad():
#             text_outputs = self.text_encoder(**text_encoding)
#             text_features = text_outputs.last_hidden_state[:, 0, :]  # Take [CLS] token
#
#         return text_features
#
#     def forward(self, image_embeddings: torch.Tensor, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Get device
#         device = image_embeddings.device
#
#         # Encode text
#         text_features = self.encode_text(text, device)
#
#         # Project features
#         projected_image = self.image_projection(image_embeddings)
#         projected_text = self.text_projection(text_features)
#
#         return projected_image, projected_text
