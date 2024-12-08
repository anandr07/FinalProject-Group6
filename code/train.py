# # train.py -biogpt
# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import logging
# from pathlib import Path
#
# from data2 import data_processing
# from alignment_model import ImageTextAlignmentModel
# from report_generator import MedicalReportGenerator
# from biovil_t.model import ImageModel
# from biovil_t.pretrained import get_biovil_t_image_encoder
#
# def train_model(csv_path: str, num_epochs: int = 30):
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Initialize models
#     image_encoder = get_biovil_t_image_encoder()
#     alignment_model = ImageTextAlignmentModel(image_embedding_dim=512)
#     report_generator = MedicalReportGenerator()
#
#     # Move models to device
#     image_encoder = image_encoder.to(device)
#     alignment_model = alignment_model.to(device)
#     report_generator = report_generator.to(device)
#
#     # Get dataloaders
#     train_loader, val_loader = data_processing.get_dataloaders(csv_path)
#
#     # Optimizers
#     alignment_optimizer = AdamW(alignment_model.parameters(), lr=2e-5)
#     # For report_generator, only optimize the PEFT parameters and the input_projection layer
#     # train.py
#     peft_params = [p for p in report_generator.model.parameters() if p.requires_grad]
#     generator_optimizer = AdamW([
#         {'params': peft_params, 'lr': 2e-5},
#         {'params': report_generator.input_projection.parameters(), 'lr': 1e-4}
#     ])
#
#     # Loss function for alignment
#     contrastive_loss = nn.CosineEmbeddingLoss()
#
#     # Training loop
#     for epoch in range(num_epochs):
#         image_encoder.eval()  # Keep image encoder in eval mode
#         alignment_model.train()
#         report_generator.train()
#
#         # Progress bar
#         progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
#
#         for batch_idx, (images, impressions) in enumerate(progress_bar):
#             # Move batch to device
#             images = images.to(device)
#
#             # Get image embeddings
#             with torch.no_grad():
#                 image_embeddings = image_encoder(images).img_embedding
#
#             # Alignment phase
#             alignment_optimizer.zero_grad()
#             projected_image, projected_text = alignment_model(image_embeddings, impressions)
#
#             # Contrastive loss
#             batch_size = images.size(0)
#             labels = torch.ones(batch_size).to(device)
#             align_loss = contrastive_loss(projected_image, projected_text, labels)
#             align_loss.backward()
#             alignment_optimizer.step()
#
#             # Generation phase
#             generator_optimizer.zero_grad()
#
#             # Prepare target text
#             target_encoding = report_generator.tokenizer(
#                 impressions,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=150
#             )
#
#             # Move target_encoding to device
#             target_ids = target_encoding['input_ids'].to(device)
#             attention_mask = target_encoding['attention_mask'].to(device)
#
#             # Forward pass with target ids
#             gen_loss, logits = report_generator(projected_image.detach(), target_ids)
#
#             # Backward pass
#             gen_loss.backward()
#             generator_optimizer.step()
#
#             # Update progress bar
#             progress_bar.set_postfix({
#                 'Align Loss': f'{align_loss.item():.4f}',
#                 'Gen Loss': f'{gen_loss.item():.4f}'
#             })
#
#             # Print sample outputs every 50 batches
#             if batch_idx % 50 == 0:
#                 with torch.no_grad():
#                     sample_report = report_generator.generate_report(projected_image[0:1].detach())[0]
#                     print("\nSample Generation:")
#                     print(f"Generated: {sample_report}")
#                     print(f"Target: {impressions[0]}\n")
#
#         # Save checkpoints at the end of each epoch
#         checkpoint_dir = Path("checkpoints")
#         checkpoint_dir.mkdir(exist_ok=True)
#
#         checkpoint = {
#             'epoch': epoch,
#             'alignment_model_state_dict': alignment_model.state_dict(),
#             'report_generator_state_dict': report_generator.state_dict(),
#             'alignment_optimizer_state_dict': alignment_optimizer.state_dict(),
#             'generator_optimizer_state_dict': generator_optimizer.state_dict(),
#             'align_loss': align_loss.item(),
#             'gen_loss': gen_loss.item()
#         }
#
#         torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
#
#         # Validation phase
#         alignment_model.eval()
#         report_generator.eval()
#
#         val_align_losses = []
#         val_gen_losses = []
#
#         with torch.no_grad():
#             for val_images, val_impressions in val_loader:
#                 val_images = val_images.to(device)
#
#                 # Get image embeddings
#                 val_image_embeddings = image_encoder(val_images).img_embedding
#
#                 # Get projections
#                 val_projected_image, val_projected_text = alignment_model(val_image_embeddings, val_impressions)
#
#                 # Calculate alignment loss
#                 val_labels = torch.ones(val_images.size(0)).to(device)
#                 val_align_loss = contrastive_loss(val_projected_image, val_projected_text, val_labels)
#
#                 # Prepare target text for generation loss
#                 val_target_encoding = report_generator.tokenizer(
#                     val_impressions,
#                     padding=True,
#                     truncation=True,
#                     return_tensors="pt",
#                     max_length=150
#                 )
#                 val_target_ids = val_target_encoding['input_ids'].to(device)
#
#                 # Calculate generation loss
#                 val_gen_loss, _ = report_generator(val_projected_image, val_target_ids)
#
#                 val_align_losses.append(val_align_loss.item())
#                 val_gen_losses.append(val_gen_loss.item())
#
#         # Calculate average validation losses
#         avg_val_align_loss = sum(val_align_losses) / len(val_align_losses)
#         avg_val_gen_loss = sum(val_gen_losses) / len(val_gen_losses)
#
#         print(f"\nEpoch {epoch+1} Validation Metrics:")
#         print(f"Avg Alignment Loss: {avg_val_align_loss:.4f}")
#         print(f"Avg Generation Loss: {avg_val_gen_loss:.4f}")
#
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     csv_path = "Data/new_csv_17k_rows.csv"
#     train_model(csv_path)




# Trail - Cite
# train.py
#
# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# import logging
# from pathlib import Path
# from tqdm import tqdm
#
# from data2 import data_processing
# from alignment_model import ImageTextEmbeddingModel
# from report_generator import MedicalReportGenerator
#
# def train_model(csv_path: str, num_epochs: int = 10):
#   # Set up logging
#   logging.basicConfig(level=logging.INFO)
#
#   # Set device
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   print(f"Using device: {device}")
#
#   # Initialize models
#   embedding_model = ImageTextEmbeddingModel()
#   report_generator = MedicalReportGenerator(input_embedding_dim=embedding_model.embedding_dim)
#
#   # Move models to device
#   embedding_model = embedding_model.to(device)
#   report_generator = report_generator.to(device)
#
#   # Get dataloaders
#   train_loader, val_loader = data_processing.get_dataloaders(csv_path)
#
#   # Collect parameters that require gradients for the embedding model
#   embedding_params = [p for p in embedding_model.parameters() if p.requires_grad]
#
#   # Optimizers
#   embedding_optimizer = AdamW(embedding_params, lr=2e-5)
#   # For report_generator, use PEFT parameters and input_projection layer
#   peft_params = [p for p in report_generator.model.parameters() if p.requires_grad]
#   generator_optimizer = AdamW([
#       {'params': peft_params, 'lr': 2e-5},
#       {'params': report_generator.input_projection.parameters(), 'lr': 1e-4}
#   ])
#
#   # Training loop
#   for epoch in range(num_epochs):
#       embedding_model.train()
#       report_generator.train()
#
#       # Progress bar
#       progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
#
#       for batch_idx, (images, impressions) in enumerate(progress_bar):
#           # Move batch to device
#           images = images.to(device)
#
#           # Get embeddings from images using Rad-DINO model
#           image_embeddings = embedding_model(images)
#
#           # Generation phase
#           generator_optimizer.zero_grad()
#           embedding_optimizer.zero_grad()
#
#           # Prepare target text
#           target_encoding = report_generator.tokenizer(
#               impressions,
#               padding=True,
#               truncation=True,
#               return_tensors="pt",
#               max_length=150
#           )
#
#           # Move target_encoding to device
#           target_ids = target_encoding['input_ids'].to(device)
#
#           # Forward pass with target ids
#           gen_loss, logits = report_generator(image_embeddings, target_ids)
#
#           # Backward pass
#           gen_loss.backward()
#           generator_optimizer.step()
#           embedding_optimizer.step()
#
#           # Update progress bar
#           progress_bar.set_postfix({
#               'Gen Loss': f'{gen_loss.item():.4f}'
#           })
#
#           # Print sample outputs every 50 batches
#           if batch_idx % 50 == 0:
#               with torch.no_grad():
#                   sample_report = report_generator.generate_report(image_embeddings[0:1])[0]
#                   print("\nSample Generation:")
#                   print(f"Generated: {sample_report}")
#                   print(f"Target: {impressions[0]}\n")
#
#       # Save checkpoints at the end of each epoch
#       checkpoint_dir = Path("checkpoints")
#       checkpoint_dir.mkdir(exist_ok=True)
#
#       checkpoint = {
#           'epoch': epoch + 1,
#           'embedding_model_state_dict': embedding_model.state_dict(),
#           'report_generator_state_dict': report_generator.state_dict(),
#           'embedding_optimizer_state_dict': embedding_optimizer.state_dict(),
#           'generator_optimizer_state_dict': generator_optimizer.state_dict(),
#           'gen_loss': gen_loss.item()
#       }
#
#       torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')
#
#       # Validation phase
#       embedding_model.eval()
#       report_generator.eval()
#
#       val_gen_losses = []
#
#       with torch.no_grad():
#           for val_images, val_impressions in val_loader:
#               val_images = val_images.to(device)
#
#               # Get embeddings from images
#               val_image_embeddings = embedding_model(val_images)
#
#               # Prepare target text for generation loss
#               val_target_encoding = report_generator.tokenizer(
#                   val_impressions,
#                   padding=True,
#                   truncation=True,
#                   return_tensors="pt",
#                   max_length=150
#               )
#               val_target_ids = val_target_encoding['input_ids'].to(device)
#
#               # Calculate generation loss
#               val_gen_loss, _ = report_generator(val_image_embeddings, val_target_ids)
#
#               val_gen_losses.append(val_gen_loss.item())
#
#       # Calculate average validation loss
#       avg_val_gen_loss = sum(val_gen_losses) / len(val_gen_losses)
#
#       print(f"\nEpoch {epoch+1} Validation Metrics:")
#       print(f"Avg Generation Loss: {avg_val_gen_loss:.4f}")
#
# if __name__ == "__main__":
#   csv_path = "Data/new_csv_17k_rows.csv"  # Adjust the path to your CSV file
#   num_epochs = 10  # Set the number of epochs
#   train_model(csv_path, num_epochs=num_epochs)

# Prompt - BioGPT
#
# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import logging
# from pathlib import Path
#
# from data2 import data_processing
# from alignment_model import ImageTextAlignmentModel
# from report_generator import MedicalReportGenerator
# from biovil_t.model import ImageModel
# from biovil_t.pretrained import get_biovil_t_image_encoder
#
# from torchmetrics.text.bleu import BLEUScore
# from torchmetrics.text.rouge import ROUGEScore
#
# def train_model(
#     csv_with_image_paths: str,
#     csv_with_labels: str,
#     num_epochs: int = 30
# ):
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Initialize models
#     image_encoder = get_biovil_t_image_encoder()
#     alignment_model = ImageTextAlignmentModel(image_embedding_dim=512)
#     report_generator = MedicalReportGenerator()
#
#     # Move models to device
#     image_encoder = image_encoder.to(device)
#     alignment_model = alignment_model.to(device)
#     report_generator = report_generator.to(device)
#
#     # Get dataloaders
#     train_loader, val_loader = data_processing.get_dataloaders(csv_with_image_paths, csv_with_labels)
#
#     # Optimizers
#     alignment_optimizer = AdamW(alignment_model.parameters(), lr=2e-5)
#     # For report_generator, only optimize the PEFT parameters and the input_projection layer
#     peft_params = [p for p in report_generator.model.parameters() if p.requires_grad]
#     generator_optimizer = AdamW([
#         {'params': peft_params, 'lr': 2e-5},
#         {'params': report_generator.input_projection.parameters(), 'lr': 1e-4}
#     ])
#
#     # Loss function for alignment
#     contrastive_loss = nn.CosineEmbeddingLoss()
#
#     # Metrics
#     bleu = BLEUScore()
#     rouge = ROUGEScore()
#
#     # Training loop
#     for epoch in range(num_epochs):
#         image_encoder.eval()  # Keep image encoder in eval mode
#         alignment_model.train()
#         report_generator.train()
#
#         # Progress bar
#         progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
#
#         train_align_losses = []
#         train_gen_losses = []
#
#         for batch_idx, (images, impressions, findings_list) in enumerate(progress_bar):
#             # Move batch to device
#             images = images.to(device)
#
#             # Get image embeddings
#             with torch.no_grad():
#                 image_embeddings = image_encoder(images).img_embedding
#
#             # Create prompt text by replacing <FINDINGS>
#             batch_prompts = []
#             for i in range(len(findings_list)):
#                 findings_str = ', '.join(findings_list[i]) if findings_list[i] else 'No Findings'
#                 prompt = f"Predicted Findings: {findings_str}. You are to act as a radiologist and write the finding section of a chest x-ray radiology report for this X-ray image and the given predicted findings. Write in the style of a radiologist, write one fluent text without enumeration, be concise and don’t provide explanations or reasons."
#                 batch_prompts.append(prompt)
#
#             # Alignment phase
#             alignment_optimizer.zero_grad()
#             projected_image, projected_text = alignment_model(image_embeddings, batch_prompts)
#
#             # Contrastive loss
#             batch_size = images.size(0)
#             labels = torch.ones(batch_size).to(device)
#             align_loss = contrastive_loss(projected_image, projected_text, labels)
#             align_loss.backward()
#             alignment_optimizer.step()
#
#             # Generation phase
#             generator_optimizer.zero_grad()
#
#             # Prepare the text prompts by tokenizing
#             prompt_encoding = report_generator.tokenizer(
#                 batch_prompts,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=512
#             )
#             prompt_input_ids = prompt_encoding['input_ids'].to(device)
#
#             # Prepare target text (impressions)
#             target_encoding = report_generator.tokenizer(
#                 impressions,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=150
#             )
#             target_ids = target_encoding['input_ids'].to(device)
#
#             # Forward pass with prompt input_ids and target ids
#             gen_loss, logits = report_generator(projected_image.detach(), prompt_input_ids, target_ids)
#
#             # Backward pass
#             gen_loss.backward()
#             generator_optimizer.step()
#
#             train_align_losses.append(align_loss.item())
#             train_gen_losses.append(gen_loss.item())
#
#             # Update progress bar
#             progress_bar.set_postfix({
#                 'Align Loss': f'{align_loss.item():.4f}',
#                 'Gen Loss': f'{gen_loss.item():.4f}'
#             })
#
#             # Print sample outputs every 50 batches
#             if batch_idx % 50 == 0:
#                 with torch.no_grad():
#                     sample_report = report_generator.generate_report(projected_image[0:1].detach(), prompt_input_ids[0:1])[0]
#                     print("\nSample Generation:")
#                     print(f"Generated: {sample_report}")
#                     print(f"Target: {impressions[0]}\n")
#
#         # Calculate average training losses
#         avg_train_align_loss = sum(train_align_losses) / len(train_align_losses)
#         avg_train_gen_loss = sum(train_gen_losses) / len(train_gen_losses)
#
#         # Save checkpoints at the end of each epoch
#         checkpoint_dir = Path("checkpoints")
#         checkpoint_dir.mkdir(exist_ok=True)
#
#         checkpoint = {
#             'epoch': epoch,
#             'alignment_model_state_dict': alignment_model.state_dict(),
#             'report_generator_state_dict': report_generator.state_dict(),
#             'alignment_optimizer_state_dict': alignment_optimizer.state_dict(),
#             'generator_optimizer_state_dict': generator_optimizer.state_dict(),
#             'align_loss': avg_train_align_loss,
#             'gen_loss': avg_train_gen_loss
#         }
#
#         torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
#
#         # Validation phase
#         alignment_model.eval()
#         report_generator.eval()
#
#         val_align_losses = []
#         val_gen_losses = []
#         val_bleu_scores = []
#         val_rouge_scores = []
#
#         with torch.no_grad():
#             for val_images, val_impressions, val_findings_list in val_loader:
#                 val_images = val_images.to(device)
#
#                 # Get image embeddings
#                 val_image_embeddings = image_encoder(val_images).img_embedding
#
#                 # Create prompts
#                 val_batch_prompts = []
#                 for i in range(len(val_findings_list)):
#                     findings_str = ', '.join(val_findings_list[i]) if val_findings_list[i] else 'No Findings'
#                     prompt = f"Predicted Findings: {findings_str}. You are to act as a radiologist and write the finding section of a chest x-ray radiology report for this X-ray image and the given predicted findings. Write in the style of a radiologist, write one fluent text without enumeration, be concise and don’t provide explanations or reasons."
#                     val_batch_prompts.append(prompt)
#
#                 # Tokenize prompts
#                 val_prompt_encoding = report_generator.tokenizer(
#                     val_batch_prompts,
#                     padding=True,
#                     truncation=True,
#                     return_tensors="pt",
#                     max_length=512
#                 )
#                 val_prompt_input_ids = val_prompt_encoding['input_ids'].to(device)
#
#                 # Get projections
#                 val_projected_image, val_projected_text = alignment_model(val_image_embeddings, val_batch_prompts)
#
#                 # Calculate alignment loss
#                 val_labels = torch.ones(val_images.size(0)).to(device)
#                 val_align_loss = contrastive_loss(val_projected_image, val_projected_text, val_labels)
#
#                 # Prepare target text for generation loss
#                 val_target_encoding = report_generator.tokenizer(
#                     val_impressions,
#                     padding=True,
#                     truncation=True,
#                     return_tensors="pt",
#                     max_length=150
#                 )
#                 val_target_ids = val_target_encoding['input_ids'].to(device)
#
#                 # Calculate generation loss
#                 val_gen_loss, _ = report_generator(val_projected_image, val_prompt_input_ids, val_target_ids)
#
#                 val_align_losses.append(val_align_loss.item())
#                 val_gen_losses.append(val_gen_loss.item())
#
#                 # Generate reports
#                 generated_reports = report_generator.generate_report(val_projected_image, val_prompt_input_ids)
#
#                 # Compute BLEU and ROUGE scores
#                 for gen_report, target_report in zip(generated_reports, val_impressions):
#                     # Tokenize the generated and target reports
#                     gen_tokens = report_generator.tokenizer.tokenize(gen_report)
#                     target_tokens = report_generator.tokenizer.tokenize(target_report)
#
#                     # BLEU score
#                     bleu_score = bleu([gen_tokens], [[target_tokens]])
#                     val_bleu_scores.append(bleu_score.item())
#
#                     # ROUGE score
#                     rouge_score = rouge(gen_report, target_report)
#                     val_rouge_scores.append(rouge_score['rougeL_fmeasure'].item())
#
#         # Calculate average validation losses and scores
#         avg_val_align_loss = sum(val_align_losses) / len(val_align_losses)
#         avg_val_gen_loss = sum(val_gen_losses) / len(val_gen_losses)
#         avg_val_bleu_score = sum(val_bleu_scores) / len(val_bleu_scores)
#         avg_val_rouge_score = sum(val_rouge_scores) / len(val_rouge_scores)
#
#         print(f"\nEpoch {epoch+1} Training Metrics:")
#         print(f"Avg Training Alignment Loss: {avg_train_align_loss:.4f}")
#         print(f"Avg Training Generation Loss: {avg_train_gen_loss:.4f}")
#
#         print(f"\nEpoch {epoch+1} Validation Metrics:")
#         print(f"Avg Validation Alignment Loss: {avg_val_align_loss:.4f}")
#         print(f"Avg Validation Generation Loss: {avg_val_gen_loss:.4f}")
#         print(f"Avg BLEU Score: {avg_val_bleu_score:.4f}")
#         print(f"Avg ROUGE Score: {avg_val_rouge_score:.4f}")
#
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     csv_with_image_paths = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/final.csv"
#     csv_with_labels = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/labeled_reports_with_images.csv"
#     train_model(csv_with_image_paths, csv_with_labels)

# train.py:

# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import logging
# from pathlib import Path
# from typing import Tuple
#
# from data2 import data_processing
# from alignment_model import ImageTextAlignmentModel
# from report_generator import MedicalReportGenerator
# from biovil_t.pretrained import get_biovil_t_image_encoder
#
# from torchmetrics.text.bleu import BLEUScore
# from torchmetrics.text.rouge import ROUGEScore
#
#
# def custom_collate_fn(batch):
#     images = torch.stack([item[0] for item in batch])  # Stack images into a batch tensor
#     impressions = [item[1] for item in batch]          # Collect impressions into a list
#     findings_list = [item[2] for item in batch]        # Collect findings_list into a list
#     return images, impressions, findings_list
#
#
# def train_model(
#     csv_with_image_paths: str,
#     csv_with_labels: str,
#     num_epochs: int = 30,
#     batch_size: int = 8,
#     train_split: float = 0.85,
#     num_workers: int = 4,
#     seed: int = 42
# ):
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Initialize models
#     image_encoder = get_biovil_t_image_encoder()
#     alignment_model = ImageTextAlignmentModel(image_embedding_dim=512)
#     report_generator = MedicalReportGenerator()
#
#     # Move models to device
#     image_encoder = image_encoder.to(device)
#     alignment_model = alignment_model.to(device)
#     report_generator = report_generator.to(device)
#
#     # Get dataloaders
#     train_loader, val_loader = data_processing.get_dataloaders(
#         csv_with_image_paths,
#         csv_with_labels,
#         batch_size=batch_size,
#         train_split=train_split,
#         num_workers=num_workers,
#         seed=seed,
#         collate_fn=custom_collate_fn  # Pass the custom collate function
#     )
#
#     # Optimizers
#     alignment_optimizer = AdamW(alignment_model.parameters(), lr=2e-5)
#     # For report_generator, only optimize the PEFT parameters and the input_projection layer
#     peft_params = [p for p in report_generator.model.parameters() if p.requires_grad]
#     generator_optimizer = AdamW([
#         {'params': peft_params, 'lr': 2e-5},
#         {'params': report_generator.input_projection.parameters(), 'lr': 1e-4}
#     ])
#
#     # Loss function for alignment
#     contrastive_loss = nn.CosineEmbeddingLoss()
#
#     # Metrics
#     bleu = BLEUScore()
#     rouge = ROUGEScore()
#
#     # Training loop
#     for epoch in range(num_epochs):
#         image_encoder.eval()  # Keep image encoder in eval mode
#         alignment_model.train()
#         report_generator.train()
#
#         # Progress bar
#         progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
#
#         train_align_losses = []
#         train_gen_losses = []
#
#         for batch_idx, (images, impressions, findings_list) in enumerate(progress_bar):
#             # Move batch to device
#             images = images.to(device)
#
#             # Get image embeddings
#             with torch.no_grad():
#                 image_embeddings = image_encoder(images).img_embedding
#
#             # Create prompt text by replacing <FINDINGS>
#             batch_prompts = []
#             for i in range(len(findings_list)):
#                 findings_str = ', '.join(findings_list[i]) if findings_list[i] else 'No Findings'
#                 prompt = f"Predicted Findings: {findings_str}. You are to act as a radiologist and write the finding section of a chest x-ray radiology report for this X-ray image and the given predicted findings. Write in the style of a radiologist, write one fluent text without enumeration, be concise and don’t provide explanations or reasons."
#                 batch_prompts.append(prompt)
#
#             # Alignment phase
#             alignment_optimizer.zero_grad()
#             projected_image, projected_text = alignment_model(image_embeddings, batch_prompts)
#
#             # Contrastive loss
#             batch_size = images.size(0)
#             labels = torch.ones(batch_size).to(device)
#             align_loss = contrastive_loss(projected_image, projected_text, labels)
#             align_loss.backward()
#             alignment_optimizer.step()
#
#             # Generation phase
#             generator_optimizer.zero_grad()
#
#             # Prepare the text prompts by tokenizing
#             prompt_encoding = report_generator.tokenizer(
#                 batch_prompts,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=512
#             )
#             prompt_input_ids = prompt_encoding['input_ids'].to(device)
#
#             # Prepare target text (impressions)
#             target_encoding = report_generator.tokenizer(
#                 impressions,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=150
#             )
#             target_ids = target_encoding['input_ids'].to(device)
#
#             # Forward pass with prompt input_ids and target ids
#             gen_loss, logits = report_generator(projected_image.detach(), prompt_input_ids, target_ids)
#
#             # Backward pass
#             gen_loss.backward()
#             generator_optimizer.step()
#
#             train_align_losses.append(align_loss.item())
#             train_gen_losses.append(gen_loss.item())
#
#             # Update progress bar
#             progress_bar.set_postfix({
#                 'Align Loss': f'{align_loss.item():.4f}',
#                 'Gen Loss': f'{gen_loss.item():.4f}'
#             })
#
#             # Print sample outputs every 50 batches
#             if batch_idx % 50 == 0:
#                 with torch.no_grad():
#                     sample_report = report_generator.generate_report(projected_image[0:1].detach(), prompt_input_ids[0:1])[0]
#                     print("\nSample Generation:")
#                     print(f"Generated: {sample_report}")
#                     print(f"Target: {impressions[0]}\n")
#
#         # Calculate average training losses
#         avg_train_align_loss = sum(train_align_losses) / len(train_align_losses)
#         avg_train_gen_loss = sum(train_gen_losses) / len(train_gen_losses)
#
#         # Save checkpoints at the end of each epoch
#         checkpoint_dir = Path("checkpoints")
#         checkpoint_dir.mkdir(exist_ok=True)
#
#         checkpoint = {
#             'epoch': epoch,
#             'alignment_model_state_dict': alignment_model.state_dict(),
#             'report_generator_state_dict': report_generator.state_dict(),
#             'alignment_optimizer_state_dict': alignment_optimizer.state_dict(),
#             'generator_optimizer_state_dict': generator_optimizer.state_dict(),
#             'align_loss': avg_train_align_loss,
#             'gen_loss': avg_train_gen_loss
#         }
#
#         torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
#
#         # Validation phase
#         alignment_model.eval()
#         report_generator.eval()
#
#         val_align_losses = []
#         val_gen_losses = []
#         val_bleu_scores = []
#         val_rouge_scores = []
#
#         with torch.no_grad():
#             for val_images, val_impressions, val_findings_list in val_loader:
#                 val_images = val_images.to(device)
#
#                 # Get image embeddings
#                 val_image_embeddings = image_encoder(val_images).img_embedding
#
#                 # Create prompts
#                 val_batch_prompts = []
#                 for i in range(len(val_findings_list)):
#                     findings_str = ', '.join(val_findings_list[i]) if val_findings_list[i] else 'No Findings'
#                     prompt = f"Predicted Findings: {findings_str}. You are to act as a radiologist and write the finding section of a chest x-ray radiology report for this X-ray image and the given predicted findings. Write in the style of a radiologist, write one fluent text without enumeration, be concise and don’t provide explanations or reasons."
#                     val_batch_prompts.append(prompt)
#
#                 # Tokenize prompts
#                 val_prompt_encoding = report_generator.tokenizer(
#                     val_batch_prompts,
#                     padding=True,
#                     truncation=True,
#                     return_tensors="pt",
#                     max_length=512
#                 )
#                 val_prompt_input_ids = val_prompt_encoding['input_ids'].to(device)
#
#                 # Get projections
#                 val_projected_image, val_projected_text = alignment_model(val_image_embeddings, val_batch_prompts)
#
#                 # Calculate alignment loss
#                 val_labels = torch.ones(val_images.size(0)).to(device)
#                 val_align_loss = contrastive_loss(val_projected_image, val_projected_text, val_labels)
#
#                 # Prepare target text for generation loss
#                 val_target_encoding = report_generator.tokenizer(
#                     val_impressions,
#                     padding=True,
#                     truncation=True,
#                     return_tensors="pt",
#                     max_length=150
#                 )
#                 val_target_ids = val_target_encoding['input_ids'].to(device)
#
#                 # Calculate generation loss
#                 val_gen_loss, _ = report_generator(val_projected_image, val_prompt_input_ids, val_target_ids)
#
#                 val_align_losses.append(val_align_loss.item())
#                 val_gen_losses.append(val_gen_loss.item())
#
#                 # Generate reports
#                 generated_reports = report_generator.generate_report(val_projected_image, val_prompt_input_ids)
#
#                 # Compute BLEU and ROUGE scores
#                 for gen_report, target_report in zip(generated_reports, val_impressions):
#                     # BLEU score
#                     bleu_score = bleu([gen_report], [[target_report]])
#                     val_bleu_scores.append(bleu_score.item())
#
#                     # ROUGE score
#                     rouge_result = rouge(gen_report, target_report)
#                     val_rouge_scores.append(rouge_result['rougeL_fmeasure'].item())
#
#         # Calculate average validation losses and scores
#         avg_val_align_loss = sum(val_align_losses) / len(val_align_losses)
#         avg_val_gen_loss = sum(val_gen_losses) / len(val_gen_losses)
#         avg_val_bleu_score = sum(val_bleu_scores) / len(val_bleu_scores)
#         avg_val_rouge_score = sum(val_rouge_scores) / len(val_rouge_scores)
#
#         print(f"\nEpoch {epoch+1} Training Metrics:")
#         print(f"Avg Training Alignment Loss: {avg_train_align_loss:.4f}")
#         print(f"Avg Training Generation Loss: {avg_train_gen_loss:.4f}")
#
#         print(f"\nEpoch {epoch+1} Validation Metrics:")
#         print(f"Avg Validation Alignment Loss: {avg_val_align_loss:.4f}")
#         print(f"Avg Validation Generation Loss: {avg_val_gen_loss:.4f}")
#         print(f"Avg BLEU Score: {avg_val_bleu_score:.4f}")
#         print(f"Avg ROUGE Score: {avg_val_rouge_score:.4f}")
#
# if __name__ == "__main__":
#     import argparse
#
#     logging.basicConfig(level=logging.INFO)
#     parser = argparse.ArgumentParser(description='Train Medical Report Generation Model')
#     parser.add_argument('--csv_with_image_paths', type=str, required=True, help='Path to CSV file with image paths')
#     parser.add_argument('--csv_with_labels', type=str, required=True, help='Path to CSV file with labels and reports')
#     parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
#     args = parser.parse_args()
#
#     train_model(
#         csv_with_image_paths=args.csv_with_image_paths,
#         csv_with_labels=args.csv_with_labels,
#         num_epochs=args.num_epochs
#     )
# train.py: - prompt included but no output
#
# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import logging
# from pathlib import Path
# from typing import Tuple
# import nltk
# nltk.download('punkt')
#
# from data2 import data_processing   # Ensure correct import
# from alignment_model import ImageTextAlignmentModel
# from report_generator import MedicalReportGenerator
# from biovil_t.pretrained import get_biovil_t_image_encoder
#
# from torchmetrics.text.bleu import BLEUScore
# from torchmetrics.text.rouge import ROUGEScore
#
#
# def custom_collate_fn(batch):
#     images = torch.stack([item[0] for item in batch])  # Stack images into a batch tensor
#     impressions = [item[1] for item in batch]          # Collect impressions into a list
#     findings_list = [item[2] for item in batch]        # Collect findings_list into a list
#     return images, impressions, findings_list
#
#
# def train_model(
#     csv_with_image_paths: str,
#     csv_with_labels: str,
#     num_epochs: int = 30,
#     batch_size: int = 8,
#     train_split: float = 0.85,
#     num_workers: int = 4,
#     seed: int = 42
# ):
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Initialize models
#     image_encoder = get_biovil_t_image_encoder()
#     alignment_model = ImageTextAlignmentModel(image_embedding_dim=512)
#     report_generator = MedicalReportGenerator()
#
#     # Move models to device
#     image_encoder = image_encoder.to(device)
#     alignment_model = alignment_model.to(device)
#     report_generator = report_generator.to(device)
#
#     # Get dataloaders
#     train_loader, val_loader = data_processing.get_dataloaders(
#         csv_with_image_paths,
#         csv_with_labels,
#         batch_size=batch_size,
#         train_split=train_split,
#         num_workers=num_workers,
#         seed=seed,
#         collate_fn=custom_collate_fn  # Pass the custom collate function
#     )
#
#     # Optimizers
#     alignment_optimizer = AdamW(alignment_model.parameters(), lr=2e-5)
#     # For report_generator, only optimize the PEFT parameters and the input_projection layer
#     peft_params = [p for p in report_generator.model.parameters() if p.requires_grad]
#     generator_optimizer = AdamW([
#         {'params': peft_params, 'lr': 2e-5},
#         {'params': report_generator.input_projection.parameters(), 'lr': 1e-4}
#     ])
#
#     # Loss function for alignment
#     contrastive_loss = nn.CosineEmbeddingLoss()
#
#     # Metrics
#     bleu = BLEUScore()
#     rouge = ROUGEScore()
#
#     # Training loop
#     for epoch in range(num_epochs):
#         image_encoder.eval()  # Keep image encoder in eval mode
#         alignment_model.train()
#         report_generator.train()
#
#         # Progress bar
#         progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
#
#         train_align_losses = []
#         train_gen_losses = []
#
#         for batch_idx, (images, impressions, findings_list) in enumerate(progress_bar):
#             # Move batch to device
#             images = images.to(device)
#
#             # Get image embeddings
#             with torch.no_grad():
#                 image_embeddings = image_encoder(images).img_embedding
#
#             # Create prompt text by replacing <FINDINGS>
#             batch_prompts = []
#             for i in range(len(findings_list)):
#                 findings_str = ', '.join(findings_list[i]) if findings_list[i] else 'No Findings'
#                 prompt = f"Predicted Findings: {findings_str}. You are to act as a radiologist and write the finding section of a chest x-ray radiology report for this X-ray image and the given predicted findings. Write in the style of a radiologist, write one fluent text without enumeration, be concise and don’t provide explanations or reasons."
#                 batch_prompts.append(prompt)
#
#             # Alignment phase
#             alignment_optimizer.zero_grad()
#             projected_image, projected_text = alignment_model(image_embeddings, batch_prompts)
#
#             # Contrastive loss
#             batch_size = images.size(0)
#             labels = torch.ones(batch_size).to(device)
#             align_loss = contrastive_loss(projected_image, projected_text, labels)
#             align_loss.backward()
#             alignment_optimizer.step()
#
#             # Generation phase
#             generator_optimizer.zero_grad()
#
#             # Prepare the text prompts by tokenizing
#             prompt_encoding = report_generator.tokenizer(
#                 batch_prompts,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=512
#             )
#             prompt_input_ids = prompt_encoding['input_ids'].to(device)
#
#             # Prepare target text (impressions)
#             target_encoding = report_generator.tokenizer(
#                 impressions,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=150
#             )
#             target_ids = target_encoding['input_ids'].to(device)
#
#             # Forward pass with prompt input_ids and target ids
#             gen_loss, logits = report_generator(projected_image.detach(), prompt_input_ids, target_ids)
#
#             # Backward pass
#             gen_loss.backward()
#             generator_optimizer.step()
#
#             train_align_losses.append(align_loss.item())
#             train_gen_losses.append(gen_loss.item())
#
#             # Update progress bar
#             progress_bar.set_postfix({
#                 'Align Loss': f'{align_loss.item():.4f}',
#                 'Gen Loss': f'{gen_loss.item():.4f}'
#             })
#
#             # Print sample outputs every 50 batches
#             if batch_idx % 50 == 0:
#                 with torch.no_grad():
#                     sample_report = report_generator.generate_report(projected_image[0:1].detach(), prompt_input_ids[0:1])[0]
#                     print("\nSample Generation:")
#                     print(f"Generated: {sample_report}")
#                     print(f"Target: {impressions[0]}\n")
#
#         # Calculate average training losses
#         avg_train_align_loss = sum(train_align_losses) / len(train_align_losses)
#         avg_train_gen_loss = sum(train_gen_losses) / len(train_gen_losses)
#
#         # Save checkpoints at the end of each epoch
#         checkpoint_dir = Path("checkpoints")
#         checkpoint_dir.mkdir(exist_ok=True)
#
#         checkpoint = {
#             'epoch': epoch,
#             'alignment_model_state_dict': alignment_model.state_dict(),
#             'report_generator_state_dict': report_generator.state_dict(),
#             'alignment_optimizer_state_dict': alignment_optimizer.state_dict(),
#             'generator_optimizer_state_dict': generator_optimizer.state_dict(),
#             'align_loss': avg_train_align_loss,
#             'gen_loss': avg_train_gen_loss
#         }
#
#         torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
#
#         # Validation phase
#         alignment_model.eval()
#         report_generator.eval()
#
#         val_align_losses = []
#         val_gen_losses = []
#         val_bleu_scores = []
#         val_rouge_scores = []
#
#         with torch.no_grad():
#             for val_images, val_impressions, val_findings_list in val_loader:
#                 val_images = val_images.to(device)
#
#                 # Get image embeddings
#                 val_image_embeddings = image_encoder(val_images).img_embedding
#
#                 # Create prompts
#                 val_batch_prompts = []
#                 for i in range(len(val_findings_list)):
#                     findings_str = ', '.join(val_findings_list[i]) if val_findings_list[i] else 'No Findings'
#                     prompt = f"Predicted Findings: {findings_str}. You are to act as a radiologist and write the finding section of a chest x-ray radiology report for this X-ray image and the given predicted findings. Write in the style of a radiologist, write one fluent text without enumeration, be concise and don’t provide explanations or reasons."
#                     val_batch_prompts.append(prompt)
#
#                 # Tokenize prompts
#                 val_prompt_encoding = report_generator.tokenizer(
#                     val_batch_prompts,
#                     padding=True,
#                     truncation=True,
#                     return_tensors="pt",
#                     max_length=512
#                 )
#                 val_prompt_input_ids = val_prompt_encoding['input_ids'].to(device)
#
#                 # Get projections
#                 val_projected_image, val_projected_text = alignment_model(val_image_embeddings, val_batch_prompts)
#
#                 # Calculate alignment loss
#                 val_labels = torch.ones(val_images.size(0)).to(device)
#                 val_align_loss = contrastive_loss(val_projected_image, val_projected_text, val_labels)
#
#                 # Prepare target text for generation loss
#                 val_target_encoding = report_generator.tokenizer(
#                     val_impressions,
#                     padding=True,
#                     truncation=True,
#                     return_tensors="pt",
#                     max_length=150
#                 )
#                 val_target_ids = val_target_encoding['input_ids'].to(device)
#
#                 # Calculate generation loss
#                 val_gen_loss, _ = report_generator(val_projected_image, val_prompt_input_ids, val_target_ids)
#
#                 val_align_losses.append(val_align_loss.item())
#                 val_gen_losses.append(val_gen_loss.item())
#
#                 # Generate reports
#                 generated_reports = report_generator.generate_report(val_projected_image, val_prompt_input_ids)
#
#                 # Compute BLEU and ROUGE scores
#                 for gen_report, target_report in zip(generated_reports, val_impressions):
#                     # BLEU score
#                     bleu_score = bleu([gen_report], [[target_report]])
#                     val_bleu_scores.append(bleu_score.item())
#
#                     # ROUGE score
#                     rouge_result = rouge(gen_report, target_report)
#                     val_rouge_scores.append(rouge_result['rougeL_fmeasure'].item())
#
#         # Calculate average validation losses and scores
#         avg_val_align_loss = sum(val_align_losses) / len(val_align_losses)
#         avg_val_gen_loss = sum(val_gen_losses) / len(val_gen_losses)
#         avg_val_bleu_score = sum(val_bleu_scores) / len(val_bleu_scores)
#         avg_val_rouge_score = sum(val_rouge_scores) / len(val_rouge_scores)
#
#         print(f"\nEpoch {epoch+1} Training Metrics:")
#         print(f"Avg Training Alignment Loss: {avg_train_align_loss:.4f}")
#         print(f"Avg Training Generation Loss: {avg_train_gen_loss:.4f}")
#
#         print(f"\nEpoch {epoch+1} Validation Metrics:")
#         print(f"Avg Validation Alignment Loss: {avg_val_align_loss:.4f}")
#         print(f"Avg Validation Generation Loss: {avg_val_gen_loss:.4f}")
#         print(f"Avg BLEU Score: {avg_val_bleu_score:.4f}")
#         print(f"Avg ROUGE Score: {avg_val_rouge_score:.4f}")
#
#
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     csv_with_image_paths = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/final.csv"
#     csv_with_labels = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/labeled_reports_with_images.csv"
#
#     train_model(
#         csv_with_image_paths=csv_with_image_paths,
#         csv_with_labels=csv_with_labels,
#         num_epochs=30
#     )

# Trail - c
# train.py
# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from transformers import get_linear_schedule_with_warmup
# from tqdm import tqdm
# import logging
# import wandb
# from pathlib import Path
# from typing import Tuple, Dict, Any
# import nltk
# from torch.cuda.amp import autocast, GradScaler
# import numpy as np
# # from evaluate import load
# import json
# import time
# from datetime import datetime
#
# nltk.download('punkt')
#
# from data2 import data_processing
# from alignment_model import ImageTextAlignmentModel
# from report_generator import MedicalReportGenerator
# from biovil_t.pretrained import get_biovil_t_image_encoder
# from metrics import (
#     calculate_bleu,
#     calculate_rouge,
#     calculate_bertscore,
#     Clinical_BERT_Scorer
# )
#
#
# class AverageMeter:
#     """Computes and stores the average and current value"""
#
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#
# class EarlyStopping:
#     """Early stopping to prevent overfitting"""
#
#     def __init__(self, patience=7, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False
#
#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss > self.best_loss - self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.counter = 0
#
#
# def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: Path) -> None:
#     """Save model checkpoint"""
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#
#     # Save regular checkpoint
#     checkpoint_path = checkpoint_dir / f'checkpoint_{timestamp}.pt'
#     torch.save(state, checkpoint_path)
#
#     # If this is the best model so far, save it as best.pt
#     if is_best:
#         best_path = checkpoint_dir / 'best.pt'
#         torch.save(state, best_path)
#
#
# def train_model(
#         csv_with_image_paths: str,
#         csv_with_labels: str,
#         num_epochs: int = 30,
#         batch_size: int = 8,
#         train_split: float = 0.85,
#         num_workers: int = 4,
#         seed: int = 42,
#         learning_rate: float = 2e-5,
#         warmup_steps: int = 1000,
#         gradient_accumulation_steps: int = 4,
#         max_grad_norm: float = 1.0,
#         use_wandb: bool = True,
#         checkpoint_dir: str = "checkpoints",
#         patience: int = 7,
# ):
#     # Set random seeds for reproducibility
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#
#     # Initialize wandb
#     if use_wandb:
#         wandb.init(
#             project="medical-report-generation",
#             config={
#                 "learning_rate": learning_rate,
#                 "epochs": num_epochs,
#                 "batch_size": batch_size,
#                 "warmup_steps": warmup_steps,
#                 "gradient_accumulation_steps": gradient_accumulation_steps,
#             }
#         )
#
#     # Set device and initialize models
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     image_encoder = get_biovil_t_image_encoder()
#     alignment_model = ImageTextAlignmentModel(image_embedding_dim=512)
#     report_generator = MedicalReportGenerator()
#
#     # Move models to device
#     image_encoder = image_encoder.to(device)
#     alignment_model = alignment_model.to(device)
#     report_generator = report_generator.to(device)
#
#     # Get dataloaders
#     train_loader, val_loader = data_processing.get_dataloaders(
#         csv_with_image_paths=csv_with_image_paths,
#         csv_with_labels=csv_with_labels,
#         batch_size=batch_size,
#         train_split=train_split,
#         num_workers=num_workers,
#         seed=seed,
#         collate_fn=data_processing.custom_collate_fn
#     )
#
#     # Initialize optimizers
#     alignment_optimizer = AdamW(
#         alignment_model.parameters(),
#         lr=learning_rate,
#         weight_decay=0.01
#     )
#     generator_optimizer = AdamW([
#         {'params': report_generator.model.parameters(), 'lr': learning_rate},
#         {'params': report_generator.input_projection.parameters(), 'lr': learning_rate * 10}
#     ])
#
#     # Initialize schedulers
#     num_training_steps = len(train_loader) * num_epochs
#     alignment_scheduler = get_linear_schedule_with_warmup(
#         alignment_optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=num_training_steps
#     )
#     generator_scheduler = get_linear_schedule_with_warmup(
#         generator_optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=num_training_steps
#     )
#
#     # Initialize loss functions
#     contrastive_loss = nn.CosineEmbeddingLoss()
#
#     # Initialize metrics
#     clinical_bert_scorer = Clinical_BERT_Scorer()
#
#     # Initialize gradient scaler and early stopping
#     scaler = GradScaler()
#     early_stopping = EarlyStopping(patience=patience)
#
#     # Training metrics tracking
#     best_val_score = float('-inf')
#
#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch + 1}/{num_epochs}")
#
#         # Training phase
#         train_metrics = train_epoch(
#             image_encoder=image_encoder,
#             alignment_model=alignment_model,
#             report_generator=report_generator,
#             train_loader=train_loader,
#             contrastive_loss=contrastive_loss,
#             alignment_optimizer=alignment_optimizer,
#             generator_optimizer=generator_optimizer,
#             alignment_scheduler=alignment_scheduler,
#             generator_scheduler=generator_scheduler,
#             scaler=scaler,
#             device=device,
#             gradient_accumulation_steps=gradient_accumulation_steps,
#             max_grad_norm=max_grad_norm
#         )
#
#         # Validation phase
#         val_metrics = validate_epoch(
#             image_encoder=image_encoder,
#             alignment_model=alignment_model,
#             report_generator=report_generator,
#             val_loader=val_loader,
#             contrastive_loss=contrastive_loss,
#             clinical_bert_scorer=clinical_bert_scorer,
#             device=device
#         )
#
#         # Log metrics
#         if use_wandb:
#             wandb.log({**train_metrics, **val_metrics})
#
#         # Save checkpoint if best model
#         is_best = val_metrics['val_bert_score'] > best_val_score
#         if is_best:
#             best_val_score = val_metrics['val_bert_score']
#
#         save_checkpoint({
#             'epoch': epoch,
#             'alignment_model_state_dict': alignment_model.state_dict(),
#             'report_generator_state_dict': report_generator.state_dict(),
#             'alignment_optimizer_state_dict': alignment_optimizer.state_dict(),
#             'generator_optimizer_state_dict': generator_optimizer.state_dict(),
#             'best_val_score': best_val_score,
#             'train_metrics': train_metrics,
#             'val_metrics': val_metrics,
#         }, is_best, Path(checkpoint_dir))
#
#         # Early stopping check
#         early_stopping(val_metrics['val_loss'])
#         if early_stopping.early_stop:
#             print("Early stopping triggered")
#             break
#
#     if use_wandb:
#         wandb.finish()
#
#
# def train_epoch(image_encoder, alignment_model, report_generator, train_loader,
#                 contrastive_loss, alignment_optimizer, generator_optimizer,
#                 alignment_scheduler, generator_scheduler, scaler, device,
#                 gradient_accumulation_steps, max_grad_norm):
#     alignment_model.train()
#     report_generator.train()
#     image_encoder.eval()
#
#     meters = {
#         'train_loss': AverageMeter(),
#         'align_loss': AverageMeter(),
#         'gen_loss': AverageMeter(),
#     }
#
#     progress_bar = tqdm(train_loader, desc='Training')
#
#     for batch_idx, (images, impressions, findings_list) in enumerate(progress_bar):
#         images = images.to(device)
#
#         # Get image embeddings
#         with torch.no_grad():
#             image_embeddings = image_encoder(images).img_embedding
#
#         # Create prompts
#         batch_prompts = [
#             f"Predicted Findings: {', '.join(findings) if findings else 'No Findings'}. "
#             f"Generate a detailed radiology report for this chest X-ray."
#             for findings in findings_list
#         ]
#
#         # Mixed precision training
#         with autocast():
#             # Alignment phase
#             projected_image, projected_text = alignment_model(image_embeddings, batch_prompts)
#
#             # Contrastive loss
#             batch_size = images.size(0)
#             labels = torch.ones(batch_size).to(device)
#             align_loss = contrastive_loss(projected_image, projected_text, labels)
#             align_loss = align_loss / gradient_accumulation_steps
#
#         # Scale and accumulate alignment gradients
#         scaler.scale(align_loss).backward()
#
#         # Generation phase
#         with autocast():
#             prompt_encoding = report_generator.tokenizer(
#                 batch_prompts,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=512
#             ).to(device)
#
#             target_encoding = report_generator.tokenizer(
#                 impressions,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=150
#             ).to(device)
#
#             gen_loss, _ = report_generator(
#                 projected_image.detach(),
#                 prompt_encoding['input_ids'],
#                 target_encoding['input_ids']
#             )
#             gen_loss = gen_loss / gradient_accumulation_steps
#
#         # Scale and accumulate generator gradients
#         scaler.scale(gen_loss).backward()
#
#         # Update metrics
#         meters['align_loss'].update(align_loss.item() * gradient_accumulation_steps)
#         meters['gen_loss'].update(gen_loss.item() * gradient_accumulation_steps)
#         meters['train_loss'].update(
#             (align_loss.item() + gen_loss.item()) * gradient_accumulation_steps
#         )
#
#         # Step optimizers and schedulers
#         if (batch_idx + 1) % gradient_accumulation_steps == 0:
#             scaler.unscale_(alignment_optimizer)
#             scaler.unscale_(generator_optimizer)
#
#             torch.nn.utils.clip_grad_norm_(
#                 alignment_model.parameters(), max_grad_norm
#             )
#             torch.nn.utils.clip_grad_norm_(
#                 report_generator.parameters(), max_grad_norm
#             )
#
#             scaler.step(alignment_optimizer)
#             scaler.step(generator_optimizer)
#             scaler.update()
#
#             alignment_optimizer.zero_grad()
#             generator_optimizer.zero_grad()
#
#             alignment_scheduler.step()
#             generator_scheduler.step()
#
#         # Update progress bar
#         progress_bar.set_postfix({
#             'align_loss': f"{meters['align_loss'].avg:.4f}",
#             'gen_loss': f"{meters['gen_loss'].avg:.4f}"
#         })
#
#     return {
#         'train_loss': meters['train_loss'].avg,
#         'train_align_loss': meters['align_loss'].avg,
#         'train_gen_loss': meters['gen_loss'].avg,
#     }
#
#
# def validate_epoch(image_encoder, alignment_model, report_generator, val_loader,
#                    contrastive_loss, clinical_bert_scorer, device):
#     alignment_model.eval()
#     report_generator.eval()
#     image_encoder.eval()
#
#     meters = {
#         'val_loss': AverageMeter(),
#         'val_align_loss': AverageMeter(),
#         'val_gen_loss': AverageMeter(),
#         'val_bert_score': AverageMeter(),
#         'val_bleu': AverageMeter(),
#         'val_rouge': AverageMeter(),
#     }
#
#     generated_reports = []
#     reference_reports = []
#
#     with torch.no_grad():
#         progress_bar = tqdm(val_loader, desc='Validation')
#
#         for images, impressions, findings_list in progress_bar:
#             images = images.to(device)
#
#             # Get image embeddings
#             image_embeddings = image_encoder(images).img_embedding
#
#             # Create prompts
#             batch_prompts = [
#                 f"Predicted Findings: {', '.join(findings) if findings else 'No Findings'}. "
#                 f"Generate a detailed radiology report for this chest X-ray."
#                 for findings in findings_list
#             ]
#
#             # Alignment phase
#             projected_image, projected_text = alignment_model(image_embeddings, batch_prompts)
#
#             # Contrastive loss
#             batch_size = images.size(0)
#             labels = torch.ones(batch_size).to(device)
#             align_loss = contrastive_loss(projected_image, projected_text, labels)
#
#             # Generation phase
#             prompt_encoding = report_generator.tokenizer(
#                 batch_prompts,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=512
#             ).to(device)
#
#             target_encoding = report_generator.tokenizer(
#                 impressions,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=150
#             ).to(device)
#
#             # Generate loss
#             gen_loss, _ = report_generator(
#                 projected_image,
#                 prompt_encoding['input_ids'],
#                 target_encoding['input_ids']
#             )
#
#             # Generate reports
#             generated = report_generator.generate_report(
#                 projected_image,
#                 prompt_encoding['input_ids']
#             )
#
#             # Calculate metrics
#             bert_scores = clinical_bert_scorer.score(generated, impressions)
#             bleu_scores = calculate_bleu(generated, impressions)
#             rouge_scores = calculate_rouge(generated, impressions)
#
#             # Update meters
#             meters['val_align_loss'].update(align_loss.item(), batch_size)
#             meters['val_gen_loss'].update(gen_loss.item(), batch_size)
#             meters['val_loss'].update(align_loss.item() + gen_loss.item(), batch_size)
#             meters['val_bert_score'].update(
#                 torch.mean(bert_scores['f1']).item(), batch_size
#             )
#             meters['val_bleu'].update(bleu_scores, batch_size)
#             meters['val_rouge'].update(rouge_scores['rougeL_fmeasure'], batch_size)
#
#             # Store reports for later analysis
#             generated_reports.extend(generated)
#             reference_reports.extend(impressions)
#
#             # Update progress bar
#             progress_bar.set_postfix({
#                 'val_loss': f"{meters['val_loss'].avg:.4f}",
#                 'bert_score': f"{meters['val_bert_score'].avg:.4f}"
#             })
#
#         # Save some example predictions
#         save_predictions(generated_reports[:5], reference_reports[:5])
#
#     return {
#         'val_loss': meters['val_loss'].avg,
#         'val_align_loss': meters['val_align_loss'].avg,
#         'val_gen_loss': meters['val_gen_loss'].avg,
#         'val_bert_score': meters['val_bert_score'].avg,
#         'val_bleu': meters['val_bleu'].avg,
#         'val_rouge': meters['val_rouge'].avg
#     }
#
#
# def save_predictions(generated_reports: list[str], reference_reports: list[str],
#                      filename: str = "example_predictions.txt"):
#     """Save example predictions for analysis"""
#     with open(filename, 'w') as f:
#         for i, (gen, ref) in enumerate(zip(generated_reports, reference_reports)):
#             f.write(f"Example {i + 1}:\n")
#             f.write(f"Generated: {gen}\n")
#             f.write(f"Reference: {ref}\n")
#             f.write("-" * 80 + "\n")
#
#
# def calculate_rouge(generated: list[str], reference: list[str]) -> Dict[str, float]:
#     """Calculate ROUGE scores for generated reports"""
#     from rouge_score import rouge_scorer
#
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     scores = {
#         'rouge1_precision': 0.0,
#         'rouge1_recall': 0.0,
#         'rouge1_fmeasure': 0.0,
#         'rouge2_precision': 0.0,
#         'rouge2_recall': 0.0,
#         'rouge2_fmeasure': 0.0,
#         'rougeL_precision': 0.0,
#         'rougeL_recall': 0.0,
#         'rougeL_fmeasure': 0.0
#     }
#
#     try:
#         for gen, ref in zip(generated, reference):
#             score = scorer.score(ref, gen)
#             # Accumulate scores
#             scores['rouge1_precision'] += score['rouge1'].precision
#             scores['rouge1_recall'] += score['rouge1'].recall
#             scores['rouge1_fmeasure'] += score['rouge1'].fmeasure
#             scores['rouge2_precision'] += score['rouge2'].precision
#             scores['rouge2_recall'] += score['rouge2'].recall
#             scores['rouge2_fmeasure'] += score['rouge2'].fmeasure
#             scores['rougeL_precision'] += score['rougeL'].precision
#             scores['rougeL_recall'] += score['rougeL'].recall
#             scores['rougeL_fmeasure'] += score['rougeL'].fmeasure
#
#         # Average the scores
#         n = len(generated)
#         if n > 0:
#             for k in scores:
#                 scores[k] = scores[k] / n
#
#         return scores
#
#     except Exception as e:
#         logging.warning(f"Error calculating ROUGE score: {e}")
#         return scores
#
#
# def calculate_bleu(generated: list[str], reference: list[str]) -> float:
#     """Calculate BLEU score for generated reports"""
#     from nltk.translate import bleu_score
#     from nltk.tokenize import word_tokenize
#
#     # Tokenize the sentences
#     references = [[word_tokenize(ref)] for ref in reference]
#     hypotheses = [word_tokenize(gen) for gen in generated]
#
#     # Calculate BLEU score
#     weights = (0.25, 0.25, 0.25, 0.25)  # Use default BLEU-4 weights
#
#     try:
#         return bleu_score.corpus_bleu(references, hypotheses, weights=weights)
#     except Exception as e:
#         logging.warning(f"Error calculating BLEU score: {e}")
#         return 0.0
#
# # def calculate_bleu(generated: list[str], reference: list[str]) -> float:
# #     """Calculate BLEU score for generated reports"""
# #     references = [[ref.split()] for ref in reference]
# #     hypotheses = [gen.split() for gen in generated]
# #
# #     # Calculate and return corpus BLEU
# #     try:
# #         return nltk.translate.bleu_score.corpus_bleu(references, hypotheses)
# #     except Exception as e:
# #         logging.warning(f"Error calculating BLEU score: {e}")
# #         return 0.0
# #
# #
# # def calculate_rouge(generated: list[str], reference: list[str]) -> Dict[str, float]:
# #     """Calculate ROUGE scores for generated reports"""
# #     from rouge_score import rouge_scorer
# #
# #     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
# #     scores = {
# #         'rouge1': 0.0,
# #         'rouge2': 0.0,
# #         'rougeL': 0.0,
# #         'rougeLsum': 0.0
# #     }
# #
# #     try:
# #         for gen, ref in zip(generated, reference):
# #             score = scorer.score(ref, gen)
# #             scores['rouge1'] += score['rouge1'].fmeasure
# #             scores['rouge2'] += score['rouge2'].fmeasure
# #             scores['rougeL'] += score['rougeL'].fmeasure
# #             scores['rougeLsum'] += score['rougeL'].fmeasure  # Using rougeL for rougeLsum
# #
# #         # Average the scores
# #         n = len(generated)
# #         for k in scores:
# #             scores[k] = scores[k] / n if n > 0 else 0.0
# #
# #         return scores
# #
# #     except Exception as e:
# #         logging.warning(f"Error calculating ROUGE score: {e}")
# #         return scores
#
#
#
# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
#
#     # Direct path assignments
#     csv_with_image_paths = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/final.csv"
#     csv_with_labels = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/labeled_reports_with_images.csv"
#
#     # Training configuration
#     config = {
#         'num_epochs': 30,
#         'batch_size': 8,
#         'learning_rate': 2e-5,
#         'warmup_steps': 1000,
#         'gradient_accumulation_steps': 4,
#         'use_wandb': True,
#         'checkpoint_dir': 'checkpoints',
#         'seed': 42
#     }
#
#     # Start training
#     train_model(
#         csv_with_image_paths=csv_with_image_paths,
#         csv_with_labels=csv_with_labels,
#         num_epochs=config['num_epochs'],
#         batch_size=config['batch_size'],
#         learning_rate=config['learning_rate'],
#         warmup_steps=config['warmup_steps'],
#         gradient_accumulation_steps=config['gradient_accumulation_steps'],
#         use_wandb=config['use_wandb'],
#         checkpoint_dir=config['checkpoint_dir'],
#         seed=config['seed']
#     )

# bLlue fixedd
# train.py
# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from transformers import get_linear_schedule_with_warmup
# from tqdm import tqdm
# import logging
# import wandb
# from pathlib import Path
# from typing import Dict, Any
# import nltk
# from torch.cuda.amp import autocast, GradScaler
# import time
# from datetime import datetime
# from metrics import Clinical_BERT_Scorer
#
# from data2 import data_processing
# from alignment_model import ImageTextAlignmentModel
# from report_generator import MedicalReportGenerator
# from biovil_t.pretrained import get_biovil_t_image_encoder
#
# nltk.download('punkt')
# nltk.download('stopwords')
#
#
# class AverageMeter:
#     """Computes and stores the average and current value"""
#
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#
# class EarlyStopping:
#     """Early stopping to prevent overfitting"""
#
#     def __init__(self, patience=7, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False
#
#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss > self.best_loss - self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.counter = 0
#
#
# def calculate_bleu(generated: list[str], reference: list[str]) -> float:
#     """Calculate corpus BLEU score"""
#     try:
#         references = [[ref.split()] for ref in reference]
#         hypotheses = [gen.split() for gen in generated]
#         return nltk.translate.bleu_score.corpus_bleu(references, hypotheses)
#     except Exception as e:
#         logging.warning(f"Error calculating BLEU score: {e}")
#         return 0.0
#
#
# def calculate_rouge(generated: list[str], reference: list[str]) -> Dict[str, float]:
#     """Calculate ROUGE scores"""
#     from rouge_score import rouge_scorer
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
#
#     try:
#         for gen, ref in zip(generated, reference):
#             score = scorer.score(ref, gen)
#             scores['rouge1'] += score['rouge1'].fmeasure
#             scores['rouge2'] += score['rouge2'].fmeasure
#             scores['rougeL'] += score['rougeL'].fmeasure
#
#         n = len(generated)
#         if n > 0:
#             for k in scores:
#                 scores[k] /= n
#         return scores
#     except Exception as e:
#         logging.warning(f"Error calculating ROUGE score: {e}")
#         return scores
#
#
# def save_examples(generated_reports: list[str], reference_reports: list[str],
#                   phase: str, epoch: int, batch: int = None) -> None:
#     """Save generated report examples"""
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     filename = f"examples/{phase}_epoch_{epoch}"
#     if batch is not None:
#         filename += f"_batch_{batch}"
#     filename += f"_{timestamp}.txt"
#
#     Path("examples").mkdir(exist_ok=True)
#     with open(filename, 'w') as f:
#         for i, (gen, ref) in enumerate(zip(generated_reports[:5], reference_reports[:5])):
#             f.write(f"Example {i + 1}:\n")
#             f.write(f"Generated: {gen}\n")
#             f.write(f"Reference: {ref}\n")
#             f.write("-" * 80 + "\n")
#
#
# def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: Path) -> None:
#     """Save model checkpoint"""
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#
#     # Save regular checkpoint
#     checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{state["epoch"]}_{timestamp}.pt'
#     torch.save(state, checkpoint_path)
#
#     # If this is the best model so far, save it as best.pt
#     if is_best:
#         best_path = checkpoint_dir / 'best.pt'
#         torch.save(state, best_path)
#
#
# def train_epoch(image_encoder, alignment_model, report_generator, train_loader,
#                 contrastive_loss, alignment_optimizer, generator_optimizer,
#                 alignment_scheduler, generator_scheduler, scaler, device,
#                 gradient_accumulation_steps, max_grad_norm, epoch):
#     alignment_model.train()
#     report_generator.train()
#     image_encoder.eval()
#
#     # Metrics tracking
#     meters = {
#         'train_loss': AverageMeter(),
#         'align_loss': AverageMeter(),
#         'gen_loss': AverageMeter(),
#     }
#
#     progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
#
#     for batch_idx, (images, impressions, findings_list) in enumerate(progress_bar):
#         images = images.to(device)
#
#         # Get image embeddings
#         with torch.no_grad():
#             image_embeddings = image_encoder(images).img_embedding
#
#         # Create prompts
#         batch_prompts = [
#             f"Predicted Findings: {', '.join(findings) if findings else 'No Findings'}. "
#             f"Generate a detailed radiology report for this chest X-ray."
#             for findings in findings_list
#         ]
#
#         # Mixed precision training
#         with autocast():
#             # Alignment phase
#             projected_image, projected_text = alignment_model(image_embeddings, batch_prompts)
#
#             # Contrastive loss
#             batch_size = images.size(0)
#             labels = torch.ones(batch_size).to(device)
#             align_loss = contrastive_loss(projected_image, projected_text, labels)
#             align_loss = align_loss / gradient_accumulation_steps
#
#         # Scale and accumulate alignment gradients
#         scaler.scale(align_loss).backward()
#
#         # Generation phase
#         with autocast():
#             prompt_encoding = report_generator.tokenizer(
#                 batch_prompts,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=512
#             ).to(device)
#
#             target_encoding = report_generator.tokenizer(
#                 impressions,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=150
#             ).to(device)
#
#             gen_loss, _ = report_generator(
#                 projected_image.detach(),
#                 prompt_encoding['input_ids'],
#                 target_encoding['input_ids']
#             )
#             gen_loss = gen_loss / gradient_accumulation_steps
#
#         # Scale and accumulate generator gradients
#         scaler.scale(gen_loss).backward()
#
#         # Update metrics
#         meters['align_loss'].update(align_loss.item() * gradient_accumulation_steps)
#         meters['gen_loss'].update(gen_loss.item() * gradient_accumulation_steps)
#         meters['train_loss'].update(
#             (align_loss.item() + gen_loss.item()) * gradient_accumulation_steps
#         )
#
#         # Step optimizers and schedulers
#         if (batch_idx + 1) % gradient_accumulation_steps == 0:
#             # Unscale gradients
#             scaler.unscale_(alignment_optimizer)
#             scaler.unscale_(generator_optimizer)
#
#             # Clip gradients
#             torch.nn.utils.clip_grad_norm_(
#                 alignment_model.parameters(), max_grad_norm
#             )
#             torch.nn.utils.clip_grad_norm_(
#                 report_generator.parameters(), max_grad_norm
#             )
#
#             # Step optimizers
#             scaler.step(alignment_optimizer)
#             scaler.step(generator_optimizer)
#             scaler.update()
#
#             # Zero gradients
#             alignment_optimizer.zero_grad()
#             generator_optimizer.zero_grad()
#
#             # Step schedulers
#             alignment_scheduler.step()
#             generator_scheduler.step()
#
#         # Update progress bar
#         progress_bar.set_postfix({
#             'align_loss': f"{meters['align_loss'].avg:.4f}",
#             'gen_loss': f"{meters['gen_loss'].avg:.4f}"
#         })
#
#         # Generate and save sample reports periodically
#         if batch_idx % 100 == 0:
#             with torch.no_grad():
#                 generated = report_generator.generate_report(
#                     projected_image.detach()[:5],
#                     prompt_encoding['input_ids'][:5]
#                 )
#                 save_examples(generated, impressions[:5], 'train', epoch, batch_idx)
#
#     return {
#         'train_loss': meters['train_loss'].avg,
#         'train_align_loss': meters['align_loss'].avg,
#         'train_gen_loss': meters['gen_loss'].avg,
#     }
#
#
# def validate_epoch(image_encoder, alignment_model, report_generator, val_loader,
#                    contrastive_loss, clinical_bert_scorer, device, epoch):
#     alignment_model.eval()
#     report_generator.eval()
#     image_encoder.eval()
#
#     # Metrics storage
#     total_val_loss = 0
#     total_align_loss = 0
#     total_gen_loss = 0
#     all_generated = []
#     all_references = []
#
#     with torch.no_grad():
#         progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
#
#         for batch_idx, (images, impressions, findings_list) in enumerate(progress_bar):
#             images = images.to(device)
#
#             # Get image embeddings
#             image_embeddings = image_encoder(images).img_embedding
#
#             # Create prompts
#             batch_prompts = [
#                 f"Predicted Findings: {', '.join(findings) if findings else 'No Findings'}. "
#                 f"Generate a detailed radiology report for this chest X-ray."
#                 for findings in findings_list
#             ]
#
#             # Alignment phase
#             projected_image, projected_text = alignment_model(image_embeddings, batch_prompts)
#
#             # Contrastive loss
#             batch_size = images.size(0)
#             labels = torch.ones(batch_size).to(device)
#             align_loss = contrastive_loss(projected_image, projected_text, labels)
#
#             # Generation phase
#             prompt_encoding = report_generator.tokenizer(
#                 batch_prompts,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=512
#             ).to(device)
#
#             target_encoding = report_generator.tokenizer(
#                 impressions,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=150
#             ).to(device)
#
#             # Generate loss
#             gen_loss, _ = report_generator(
#                 projected_image,
#                 prompt_encoding['input_ids'],
#                 target_encoding['input_ids']
#             )
#
#             # Generate reports
#             generated = report_generator.generate_report(
#                 projected_image,
#                 prompt_encoding['input_ids']
#             )
#
#             # Store for overall metrics calculation
#             all_generated.extend(generated)
#             all_references.extend(impressions)
#
#             # Update totals
#             total_val_loss += (align_loss.item() + gen_loss.item()) * batch_size
#             total_align_loss += align_loss.item() * batch_size
#             total_gen_loss += gen_loss.item() * batch_size
#
#             # Show sample reports every 50 batches
#             if batch_idx % 10 == 0:
#                 print(f"\nSample Generation (Batch {batch_idx}):")
#                 print(f"Generated: {generated[0]}")
#                 print(f"Reference: {impressions[0]}\n")
#
#     # Calculate overall metrics
#     num_samples = len(val_loader.dataset)
#     metrics = {
#         'val_loss': total_val_loss / num_samples,
#         'val_align_loss': total_align_loss / num_samples,
#         'val_gen_loss': total_gen_loss / num_samples,
#         'val_bleu': calculate_bleu(all_generated, all_references),
#         'val_rouge_l': calculate_rouge(all_generated, all_references)['rougeL']
#     }
#
#     # Save validation examples
#     save_examples(all_generated[:10], all_references[:10], 'validation', epoch)
#
#     # Print validation metrics
#     print(f"\nEpoch {epoch} Validation Metrics:")
#     for k, v in metrics.items():
#         print(f"{k}: {v:.4f}")
#
#     return metrics
#
#
# def train_model(
#         csv_with_image_paths: str,
#         csv_with_labels: str,
#         num_epochs: int = 30,
#         batch_size: int = 8,
#         train_split: float = 0.85,
#         num_workers: int = 4,
#         learning_rate: float = 2e-5,
#         warmup_steps: int = 1000,
#         gradient_accumulation_steps: int = 4,
#         max_grad_norm: float = 1.0,
#         use_wandb: bool = True,
#         checkpoint_dir: str = "checkpoints",
#         patience: int = 7,
#         seed: int = 42
# ):
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Initialize models
#     image_encoder = get_biovil_t_image_encoder()
#     alignment_model = ImageTextAlignmentModel(image_embedding_dim=512)
#     report_generator = MedicalReportGenerator()
#
#     # Move models to device
#     image_encoder = image_encoder.to(device)
#     alignment_model = alignment_model.to(device)
#     report_generator = report_generator.to(device)
#
#     # Initialize wandb
#     if use_wandb:
#         wandb.init(
#             project="medical-report-generation",
#             config={
#                 "learning_rate": learning_rate,
#                 "epochs": num_epochs,
#                 "batch_size": batch_size,
#                 "warmup_steps": warmup_steps,
#                 "gradient_accumulation_steps": gradient_accumulation_steps,
#             }
#         )
#
#     # Get dataloaders
#     train_loader, val_loader = data_processing.get_dataloaders(
#         csv_with_image_paths=csv_with_image_paths,
#         csv_with_labels=csv_with_labels,
#         batch_size=batch_size,
#         train_split=train_split,
#         num_workers=num_workers,
#         seed=seed,
#     )
#
#     # Initialize optimizers
#     alignment_optimizer = AdamW(
#         alignment_model.parameters(),
#         lr=learning_rate,
#         weight_decay=0.01
#     )
#     generator_optimizer = AdamW([
#         {'params': report_generator.model.parameters(), 'lr': learning_rate},
#         {'params': report_generator.input_projection.parameters(), 'lr': learning_rate * 10}
#     ])
#
#     # Initialize schedulers
#     num_training_steps = len(train_loader) * num_epochs
#     alignment_scheduler = get_linear_schedule_with_warmup(
#         alignment_optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=num_training_steps
#     )
#     generator_scheduler = get_linear_schedule_with_warmup(
#         generator_optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=num_training_steps
#     )
#
#     # Initialize loss functions and other components
#     contrastive_loss = nn.CosineEmbeddingLoss()
#     clinical_bert_scorer = Clinical_BERT_Scorer()
#     scaler = GradScaler()
#     early_stopping = EarlyStopping(patience=patience)
#
#     # Training metrics tracking
#     best_val_score = float('-inf')
#
#     # Create checkpoint directory
#     checkpoint_dir = Path(checkpoint_dir)
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#
#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch + 1}/{num_epochs}")
#
#         # Training phase
#         train_metrics = train_epoch(
#             image_encoder=image_encoder,
#             alignment_model=alignment_model,
#             report_generator=report_generator,
#             train_loader=train_loader,
#             contrastive_loss=contrastive_loss,
#             alignment_optimizer=alignment_optimizer,
#             generator_optimizer=generator_optimizer,
#             alignment_scheduler=alignment_scheduler,
#             generator_scheduler=generator_scheduler,
#             scaler=scaler,
#             device=device,
#             gradient_accumulation_steps=gradient_accumulation_steps,
#             max_grad_norm=max_grad_norm,
#             epoch=epoch + 1
#         )
#
#         # Validation phase
#         val_metrics = validate_epoch(
#             image_encoder=image_encoder,
#             alignment_model=alignment_model,
#             report_generator=report_generator,
#             val_loader=val_loader,
#             contrastive_loss=contrastive_loss,
#             clinical_bert_scorer=clinical_bert_scorer,
#             device=device,
#             epoch=epoch + 1
#         )
#
#         # Log metrics
#         if use_wandb:
#             wandb.log({**train_metrics, **val_metrics})
#
#         # Save checkpoint if best model
#         current_val_score = val_metrics['val_bleu'] + val_metrics['val_rouge_l']
#         is_best = current_val_score > best_val_score
#         if is_best:
#             best_val_score = current_val_score
#
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'alignment_model_state_dict': alignment_model.state_dict(),
#             'report_generator_state_dict': report_generator.state_dict(),
#             'alignment_optimizer_state_dict': alignment_optimizer.state_dict(),
#             'generator_optimizer_state_dict': generator_optimizer.state_dict(),
#             'alignment_scheduler_state_dict': alignment_scheduler.state_dict(),
#             'generator_scheduler_state_dict': generator_scheduler.state_dict(),
#             'best_val_score': best_val_score,
#             'train_metrics': train_metrics,
#             'val_metrics': val_metrics,
#         }, is_best, checkpoint_dir)
#
#         # Early stopping check
#         early_stopping(val_metrics['val_loss'])
#         if early_stopping.early_stop:
#             print("Early stopping triggered")
#             break
#
#     if use_wandb:
#         wandb.finish()
#
#
# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
#
#     # Direct path assignments
#     csv_with_image_paths = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/final.csv"
#     csv_with_labels = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/labeled_reports_with_images.csv"
#
#     # Training configuration
#     config = {
#         'num_epochs': 30,
#         'batch_size': 8,
#         'learning_rate': 1e-5,
#         'warmup_steps': 1000,
#         'gradient_accumulation_steps': 4,
#         'use_wandb': True,
#         'checkpoint_dir': 'checkpoints',
#         'seed': 42
#     }
#
#     # Start training
#     train_model(
#         csv_with_image_paths=csv_with_image_paths,
#         csv_with_labels=csv_with_labels,
#         **config
#     )

# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from transformers import get_linear_schedule_with_warmup
# from tqdm import tqdm
# import logging
# import wandb
# from pathlib import Path
# from typing import Dict, Any
# import nltk
# from torch.cuda.amp import autocast, GradScaler
# import time
# from datetime import datetime
# from metrics import Clinical_BERT_Scorer
#
# from data2 import data_processing
# from report_generator import MedicalReportGenerator
# from biovil_t.pretrained import get_biovil_t_image_encoder
#
# nltk.download('punkt')
# nltk.download('stopwords')
#
#
# class AverageMeter:
#     """Computes and stores the average and current value"""
#
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#
# class EarlyStopping:
#     """Early stopping to prevent overfitting"""
#
#     def __init__(self, patience=7, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False
#
#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss > self.best_loss - self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.counter = 0
#
#
# def calculate_bleu(generated: list[str], reference: list[str]) -> float:
#     """Calculate corpus BLEU score"""
#     try:
#         references = [[ref.split()] for ref in reference]
#         hypotheses = [gen.split() for gen in generated]
#         return nltk.translate.bleu_score.corpus_bleu(references, hypotheses)
#     except Exception as e:
#         logging.warning(f"Error calculating BLEU score: {e}")
#         return 0.0
#
#
# def calculate_rouge(generated: list[str], reference: list[str]) -> Dict[str, float]:
#     """Calculate ROUGE scores"""
#     from rouge_score import rouge_scorer
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
#
#     try:
#         for gen, ref in zip(generated, reference):
#             score = scorer.score(ref, gen)
#             scores['rouge1'] += score['rouge1'].fmeasure
#             scores['rouge2'] += score['rouge2'].fmeasure
#             scores['rougeL'] += score['rougeL'].fmeasure
#
#         n = len(generated)
#         if n > 0:
#             for k in scores:
#                 scores[k] /= n
#         return scores
#     except Exception as e:
#         logging.warning(f"Error calculating ROUGE score: {e}")
#         return scores
#
#
# def save_examples(generated_reports: list[str], reference_reports: list[str],
#                   phase: str, epoch: int, batch: int = None) -> None:
#     """Save generated report examples"""
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     filename = f"examples/{phase}_epoch_{epoch}"
#     if batch is not None:
#         filename += f"_batch_{batch}"
#     filename += f"_{timestamp}.txt"
#
#     Path("examples").mkdir(exist_ok=True)
#     with open(filename, 'w') as f:
#         for i, (gen, ref) in enumerate(zip(generated_reports[:5], reference_reports[:5])):
#             f.write(f"Example {i + 1}:\n")
#             f.write(f"Generated: {gen}\n")
#             f.write(f"Reference: {ref}\n")
#             f.write("-" * 80 + "\n")
#
#
# def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: Path) -> None:
#     """Save model checkpoint"""
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#
#     # Save regular checkpoint
#     checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{state["epoch"]}_{timestamp}.pt'
#     torch.save(state, checkpoint_path)
#
#     # If this is the best model so far, save it as best.pt
#     if is_best:
#         best_path = checkpoint_dir / 'best.pt'
#         torch.save(state, best_path)
#
#
# def train_epoch(image_encoder, report_generator, train_loader,
#                 generator_optimizer, generator_scheduler, scaler, device,
#                 gradient_accumulation_steps, max_grad_norm, epoch):
#     report_generator.train()
#     image_encoder.eval()
#
#     # Metrics tracking
#     meters = {
#         'train_loss': AverageMeter(),
#     }
#
#     progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
#
#     for batch_idx, (images, impressions, findings_list) in enumerate(progress_bar):
#         images = images.to(device)
#
#         # Get image embeddings
#         with torch.no_grad():
#             image_embeddings = image_encoder(images).img_embedding
#
#         # Prepare findings texts
#         findings_texts = [
#             ", ".join(findings) if findings else 'No Findings'
#             for findings in findings_list
#         ]
#
#         # Tokenize findings texts
#         findings_encoding = report_generator.tokenizer(
#             findings_texts,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=50
#         ).to(device)
#
#         # Prepare prompts
#         batch_prompts = [
#             "Generate a detailed radiology report for this chest X-ray."
#             for _ in findings_list
#         ]
#         prompt_encoding = report_generator.tokenizer(
#             batch_prompts,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=50
#         ).to(device)
#
#         # Tokenize targets (impressions)
#         target_encoding = report_generator.tokenizer(
#             impressions,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=150
#         ).to(device)
#
#         # Mixed precision training
#         with autocast():
#             gen_loss, _ = report_generator(
#                 image_embeddings,
#                 findings_input_ids=findings_encoding['input_ids'],
#                 prompt_input_ids=prompt_encoding['input_ids'],
#                 target_ids=target_encoding['input_ids']
#             )
#             gen_loss = gen_loss / gradient_accumulation_steps
#
#         # Scale and accumulate generator gradients
#         scaler.scale(gen_loss).backward()
#
#         # Update metrics
#         meters['train_loss'].update(
#             gen_loss.item() * gradient_accumulation_steps
#         )
#
#         # Step optimizer and scheduler
#         if (batch_idx + 1) % gradient_accumulation_steps == 0:
#             # Unscale gradients
#             scaler.unscale_(generator_optimizer)
#
#             # Clip gradients
#             torch.nn.utils.clip_grad_norm_(
#                 report_generator.parameters(), max_grad_norm
#             )
#
#             # Step optimizer
#             scaler.step(generator_optimizer)
#             scaler.update()
#
#             # Zero gradients
#             generator_optimizer.zero_grad()
#
#             # Step scheduler
#             generator_scheduler.step()
#
#         # Update progress bar
#         progress_bar.set_postfix({
#             'train_loss': f"{meters['train_loss'].avg:.4f}"
#         })
#
#     return {
#         'train_loss': meters['train_loss'].avg,
#     }
#
#
# def validate_epoch(image_encoder, report_generator, val_loader,
#                    clinical_bert_scorer, device, epoch):
#     report_generator.eval()
#     image_encoder.eval()
#
#     # Metrics storage
#     total_val_loss = 0
#     all_generated = []
#     all_references = []
#
#     with torch.no_grad():
#         progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
#
#         for batch_idx, (images, impressions, findings_list) in enumerate(progress_bar):
#             images = images.to(device)
#
#             # Get image embeddings
#             image_embeddings = image_encoder(images).img_embedding
#
#             # Prepare findings texts
#             findings_texts = [
#                 ", ".join(findings) if findings else 'No Findings'
#                 for findings in findings_list
#             ]
#
#             # Tokenize findings texts
#             findings_encoding = report_generator.tokenizer(
#                 findings_texts,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=50
#             ).to(device)
#
#             # Prepare prompts
#             batch_prompts = [
#                 "Generate a detailed radiology report for this chest X-ray."
#                 for _ in findings_list
#             ]
#             prompt_encoding = report_generator.tokenizer(
#                 batch_prompts,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=50
#             ).to(device)
#
#             # Tokenize targets (impressions)
#             target_encoding = report_generator.tokenizer(
#                 impressions,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=150
#             ).to(device)
#
#             # Generate loss
#             gen_loss, _ = report_generator(
#                 image_embeddings,
#                 findings_input_ids=findings_encoding['input_ids'],
#                 prompt_input_ids=prompt_encoding['input_ids'],
#                 target_ids=target_encoding['input_ids']
#             )
#
#             # Generate reports
#             generated = report_generator.generate_report(
#                 image_embeddings,
#                 findings_input_ids=findings_encoding['input_ids'],
#                 prompt_input_ids=prompt_encoding['input_ids']
#             )
#
#             # Store for overall metrics calculation
#             all_generated.extend(generated)
#             all_references.extend(impressions)
#
#             # Update total loss
#             total_val_loss += gen_loss.item() * images.size(0)
#
#             # Generate sample reports every 10 batches
#             if batch_idx % 10 == 0:
#                 print(f"\nSample Generation (Batch {batch_idx}):")
#                 print(f"Generated: {generated[0]}")
#                 print(f"Reference: {impressions[0]}\n")
#
#     # Calculate overall metrics
#     num_samples = len(val_loader.dataset)
#     metrics = {
#         'val_loss': total_val_loss / num_samples,
#         'val_bleu': calculate_bleu(all_generated, all_references),
#         'val_rouge_l': calculate_rouge(all_generated, all_references)['rougeL']
#     }
#
#     # Save validation examples
#     save_examples(all_generated[:10], all_references[:10], 'validation', epoch)
#
#     # Print validation metrics
#     print(f"\nEpoch {epoch} Validation Metrics:")
#     for k, v in metrics.items():
#         print(f"{k}: {v:.4f}")
#
#     return metrics
#
#
# def train_model(
#         csv_with_image_paths: str,
#         csv_with_labels: str,
#         num_epochs: int = 30,
#         batch_size: int = 8,
#         train_split: float = 0.85,
#         num_workers: int = 4,
#         learning_rate: float = 2e-5,
#         warmup_steps: int = 1000,
#         gradient_accumulation_steps: int = 4,
#         max_grad_norm: float = 1.0,
#         use_wandb: bool = True,
#         checkpoint_dir: str = "checkpoints",
#         patience: int = 7,
#         seed: int = 42
# ):
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Initialize models
#     image_encoder = get_biovil_t_image_encoder()
#     report_generator = MedicalReportGenerator()
#
#     # Move models to device
#     image_encoder = image_encoder.to(device)
#     report_generator = report_generator.to(device)
#
#     # Initialize wandb
#     if use_wandb:
#         wandb.init(
#             project="medical-report-generation",
#             config={
#                 "learning_rate": learning_rate,
#                 "epochs": num_epochs,
#                 "batch_size": batch_size,
#                 "warmup_steps": warmup_steps,
#                 "gradient_accumulation_steps": gradient_accumulation_steps,
#             }
#         )
#
#     # Get dataloaders
#     train_loader, val_loader = data_processing.get_dataloaders(
#         csv_with_image_paths=csv_with_image_paths,
#         csv_with_labels=csv_with_labels,
#         batch_size=batch_size,
#         train_split=train_split,
#         num_workers=num_workers,
#         seed=seed,
#     )
#
#     # Initialize optimizer
#     generator_optimizer = AdamW([
#         {'params': report_generator.model.parameters(), 'lr': learning_rate},
#         {'params': report_generator.input_projection.parameters(), 'lr': learning_rate * 10}
#     ])
#
#     # Initialize scheduler
#     num_training_steps = len(train_loader) * num_epochs
#     generator_scheduler = get_linear_schedule_with_warmup(
#         generator_optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=num_training_steps
#     )
#
#     # Initialize loss functions and other components
#     clinical_bert_scorer = Clinical_BERT_Scorer()
#     scaler = GradScaler()
#     early_stopping = EarlyStopping(patience=patience)
#
#     # Training metrics tracking
#     best_val_score = float('-inf')
#
#     # Create checkpoint directory
#     checkpoint_dir = Path(checkpoint_dir)
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#
#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch + 1}/{num_epochs}")
#
#         # Training phase
#         train_metrics = train_epoch(
#             image_encoder=image_encoder,
#             report_generator=report_generator,
#             train_loader=train_loader,
#             generator_optimizer=generator_optimizer,
#             generator_scheduler=generator_scheduler,
#             scaler=scaler,
#             device=device,
#             gradient_accumulation_steps=gradient_accumulation_steps,
#             max_grad_norm=max_grad_norm,
#             epoch=epoch + 1
#         )
#
#         # Validation phase
#         val_metrics = validate_epoch(
#             image_encoder=image_encoder,
#             report_generator=report_generator,
#             val_loader=val_loader,
#             clinical_bert_scorer=clinical_bert_scorer,
#             device=device,
#             epoch=epoch + 1
#         )
#
#         # Log metrics
#         if use_wandb:
#             wandb.log({**train_metrics, **val_metrics})
#
#         # Save checkpoint if best model
#         current_val_score = val_metrics['val_bleu'] + val_metrics['val_rouge_l']
#         is_best = current_val_score > best_val_score
#         if is_best:
#             best_val_score = current_val_score
#
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'report_generator_state_dict': report_generator.state_dict(),
#             'generator_optimizer_state_dict': generator_optimizer.state_dict(),
#             'generator_scheduler_state_dict': generator_scheduler.state_dict(),
#             'best_val_score': best_val_score,
#             'train_metrics': train_metrics,
#             'val_metrics': val_metrics,
#         }, is_best, checkpoint_dir)
#
#         # Early stopping check
#         early_stopping(val_metrics['val_loss'])
#         if early_stopping.early_stop:
#             print("Early stopping triggered")
#             break
#
#         # Display training and validation loss after each epoch
#         print(f"Epoch {epoch + 1} Training Loss: {train_metrics['train_loss']:.4f}")
#         print(f"Epoch {epoch + 1} Validation Loss: {val_metrics['val_loss']:.4f}")
#
#     if use_wandb:
#         wandb.finish()
#
#
# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
#
#     # Direct path assignments
#     csv_with_image_paths = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/final.csv"
#     csv_with_labels = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/labeled_reports_with_images.csv"
#
#     # Training configuration
#     config = {
#         'num_epochs': 30,
#         'batch_size': 8,
#         'learning_rate': 1e-5,
#         'warmup_steps': 1000,
#         'gradient_accumulation_steps': 4,
#         'use_wandb': True,
#         'checkpoint_dir': 'checkpoints',
#         'seed': 42
#     }
#
#     # Start training
#     train_model(
#         csv_with_image_paths=csv_with_image_paths,
#         csv_with_labels=csv_with_labels,
#         **config
#     )

# Working concat
# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from transformers import get_linear_schedule_with_warmup
# from tqdm import tqdm
# import logging
# import wandb
# from pathlib import Path
# from typing import Dict, Any
# import nltk
# from torch.cuda.amp import autocast, GradScaler
# import time
# from datetime import datetime
# from metrics import Clinical_BERT_Scorer
#
# from data2 import data_processing
# from alignment_model import ImageTextAlignmentModel
# from report_generator import MedicalReportGenerator
# from biovil_t.pretrained import get_biovil_t_image_encoder
#
# nltk.download('punkt')
# nltk.download('stopwords')
#
#
# class AverageMeter:
#     """Computes and stores the average and current value"""
#
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#
# class EarlyStopping:
#     """Early stopping to prevent overfitting"""
#
#     def __init__(self, patience=7, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False
#
#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss > self.best_loss - self.min_delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.counter = 0
#
#
# def calculate_bleu(generated: list[str], reference: list[str]) -> float:
#     """Calculate corpus BLEU score"""
#     try:
#         references = [[ref.split()] for ref in reference]
#         hypotheses = [gen.split() for gen in generated]
#         return nltk.translate.bleu_score.corpus_bleu(references, hypotheses)
#     except Exception as e:
#         logging.warning(f"Error calculating BLEU score: {e}")
#         return 0.0
#
#
# def calculate_rouge(generated: list[str], reference: list[str]) -> Dict[str, float]:
#     """Calculate ROUGE scores"""
#     from rouge_score import rouge_scorer
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#     scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
#
#     try:
#         for gen, ref in zip(generated, reference):
#             score = scorer.score(ref, gen)
#             scores['rouge1'] += score['rouge1'].fmeasure
#             scores['rouge2'] += score['rouge2'].fmeasure
#             scores['rougeL'] += score['rougeL'].fmeasure
#
#         n = len(generated)
#         if n > 0:
#             for k in scores:
#                 scores[k] /= n
#         return scores
#     except Exception as e:
#         logging.warning(f"Error calculating ROUGE score: {e}")
#         return scores
#
#
# def save_examples(generated_reports: list[str], reference_reports: list[str],
#                   phase: str, epoch: int, batch: int = None) -> None:
#     """Save generated report examples"""
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     filename = f"examples/{phase}_epoch_{epoch}"
#     if batch is not None:
#         filename += f"_batch_{batch}"
#     filename += f"_{timestamp}.txt"
#
#     Path("examples").mkdir(exist_ok=True)
#     with open(filename, 'w') as f:
#         for i, (gen, ref) in enumerate(zip(generated_reports[:5], reference_reports[:5])):
#             f.write(f"Example {i + 1}:\n")
#             f.write(f"Generated: {gen}\n")
#             f.write(f"Reference: {ref}\n")
#             f.write("-" * 80 + "\n")
#
#
# def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: Path) -> None:
#     """Save model checkpoint"""
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#
#     # Save regular checkpoint
#     checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{state["epoch"]}_{timestamp}.pt'
#     torch.save(state, checkpoint_path)
#
#     # If this is the best model so far, save it as best.pt
#     if is_best:
#         best_path = checkpoint_dir / 'best.pt'
#         torch.save(state, best_path)
#
#
# def train_epoch(image_encoder, alignment_model, report_generator, train_loader,
#                 generator_optimizer, generator_scheduler, scaler, device,
#                 gradient_accumulation_steps, max_grad_norm, epoch):
#     report_generator.train()
#     image_encoder.eval()
#     alignment_model.eval()
#
#     # Metrics tracking
#     meters = {
#         'train_loss': AverageMeter(),
#     }
#
#     progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
#
#     for batch_idx, (images, impressions, findings_list) in enumerate(progress_bar):
#         images = images.to(device)
#
#         # Get image embeddings
#         with torch.no_grad():
#             image_embeddings = image_encoder(images).img_embedding
#
#         # Create prompts (findings)
#         batch_prompts = []
#         for findings in findings_list:
#             if len(findings) == 0:
#                 findings_text = 'No Findings'
#             else:
#                 findings_text = ', '.join(findings)
#             batch_prompts.append(findings_text)
#
#         # Get combined embeddings
#         with torch.no_grad():
#             combined_embeddings = alignment_model(image_embeddings, batch_prompts)
#
#         # Prepare target_ids
#         target_encoding = report_generator.tokenizer(
#             impressions,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=150
#         ).to(device)
#
#         with autocast():
#             loss, _ = report_generator(
#                 combined_embeddings=combined_embeddings,
#                 target_ids=target_encoding['input_ids']
#             )
#             loss = loss / gradient_accumulation_steps
#
#         # Scale and accumulate gradients
#         scaler.scale(loss).backward()
#
#         # Update metrics
#         meters['train_loss'].update(loss.item() * gradient_accumulation_steps)
#
#         # Step optimizer and scheduler
#         if (batch_idx + 1) % gradient_accumulation_steps == 0:
#             # Unscale gradients
#             scaler.unscale_(generator_optimizer)
#
#             # Clip gradients
#             torch.nn.utils.clip_grad_norm_(
#                 report_generator.parameters(), max_grad_norm
#             )
#
#             # Step optimizer
#             scaler.step(generator_optimizer)
#             scaler.update()
#
#             # Zero gradients
#             generator_optimizer.zero_grad()
#
#             # Step scheduler
#             generator_scheduler.step()
#
#         # Update progress bar
#         progress_bar.set_postfix({
#             'train_loss': f"{meters['train_loss'].avg:.4f}",
#         })
#
#     return {
#         'train_loss': meters['train_loss'].avg,
#     }
#
#
# def validate_epoch(image_encoder, alignment_model, report_generator, val_loader,
#                    clinical_bert_scorer, device, epoch):
#     report_generator.eval()
#     image_encoder.eval()
#     alignment_model.eval()
#
#     # Metrics storage
#     total_val_loss = 0
#     all_generated = []
#     all_references = []
#
#     with torch.no_grad():
#         progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
#
#         for batch_idx, (images, impressions, findings_list) in enumerate(progress_bar):
#             images = images.to(device)
#
#             # Get image embeddings
#             image_embeddings = image_encoder(images).img_embedding
#
#             # Create prompts (findings)
#             batch_prompts = []
#             for findings in findings_list:
#                 # print(findings)
#                 if len(findings) == 0:
#                     print("Empty")
#                     findings_text = 'No Findings'
#                 else:
#                     findings_text = ', '.join(findings)
#                 batch_prompts.append(findings_text)
#
#             # Get combined embeddings
#             combined_embeddings = alignment_model(image_embeddings, batch_prompts)
#
#             # Prepare target_ids
#             target_encoding = report_generator.tokenizer(
#                 impressions,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=150
#             ).to(device)
#
#             # Calculate loss
#             loss, _ = report_generator(
#                 combined_embeddings=combined_embeddings,
#                 target_ids=target_encoding['input_ids']
#             )
#
#             # Generate reports
#             generated = report_generator.generate_report(
#                 combined_embeddings=combined_embeddings
#             )
#
#             # Store for overall metrics calculation
#             all_generated.extend(generated)
#             all_references.extend(impressions)
#
#             # Update total loss
#             total_val_loss += loss.item() * images.size(0)
#
#             # Generate sample reports every 10 batches
#             if batch_idx % 3 == 0:
#                 print(f"\nSample Generation (Batch {batch_idx}):")
#                 print(f"Findings Prompt: {batch_prompts[0]}")
#                 print(f"Generated: {generated[0]}")
#                 print(f"Reference: {impressions[0]}\n")
#
#         # Calculate overall metrics
#         num_samples = len(val_loader.dataset)
#         avg_val_loss = total_val_loss / num_samples
#         bleu_score = calculate_bleu(all_generated, all_references)
#         rouge_scores = calculate_rouge(all_generated, all_references)
#
#         metrics = {
#             'val_loss': avg_val_loss,
#             'val_bleu': bleu_score,
#             'val_rouge_l': rouge_scores['rougeL']
#         }
#
#         # Print validation metrics
#         print(f"\nEpoch {epoch} Validation Metrics:")
#         for k, v in metrics.items():
#             print(f"{k}: {v:.4f}")
#
#     return metrics
#
#
# def train_model(
#         csv_with_image_paths: str,
#         csv_with_labels: str,
#         num_epochs: int = 30,
#         batch_size: int = 8,
#         train_split: float = 0.85,
#         num_workers: int = 4,
#         learning_rate: float = 2e-5,
#         warmup_steps: int = 1000,
#         gradient_accumulation_steps: int = 4,
#         max_grad_norm: float = 1.0,
#         use_wandb: bool = True,
#         checkpoint_dir: str = "checkpoints",
#         patience: int = 7,
#         seed: int = 42
# ):
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Initialize models
#     image_encoder = get_biovil_t_image_encoder()
#     alignment_model = ImageTextAlignmentModel(image_embedding_dim=512)
#     report_generator = MedicalReportGenerator()
#
#     # Move models to device
#     image_encoder = image_encoder.to(device)
#     alignment_model = alignment_model.to(device)
#     report_generator = report_generator.to(device)
#
#     # Initialize wandb
#     if use_wandb:
#         wandb.init(
#             project="medical-report-generation",
#             config={
#                 "learning_rate": learning_rate,
#                 "epochs": num_epochs,
#                 "batch_size": batch_size,
#                 "warmup_steps": warmup_steps,
#                 "gradient_accumulation_steps": gradient_accumulation_steps,
#             }
#         )
#
#     # Get dataloaders
#     train_loader, val_loader = data_processing.get_dataloaders(
#         csv_with_image_paths=csv_with_image_paths,
#         csv_with_labels=csv_with_labels,
#         batch_size=batch_size,
#         train_split=train_split,
#         num_workers=num_workers,
#         seed=seed,
#     )
#
#     # Initialize optimizer
#     generator_optimizer = AdamW(
#         report_generator.parameters(),
#         lr=learning_rate,
#         weight_decay=0.01
#     )
#
#     # Initialize scheduler
#     num_training_steps = len(train_loader) * num_epochs
#     generator_scheduler = get_linear_schedule_with_warmup(
#         generator_optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=num_training_steps
#     )
#
#     # Initialize loss functions and other components
#     clinical_bert_scorer = Clinical_BERT_Scorer()
#     scaler = GradScaler()
#     early_stopping = EarlyStopping(patience=patience)
#
#     # Training metrics tracking
#     best_val_score = float('-inf')
#
#     # Create checkpoint directory
#     checkpoint_dir = Path(checkpoint_dir)
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#
#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch + 1}/{num_epochs}")
#
#         # Training phase
#         train_metrics = train_epoch(
#             image_encoder=image_encoder,
#             alignment_model=alignment_model,
#             report_generator=report_generator,
#             train_loader=train_loader,
#             generator_optimizer=generator_optimizer,
#             generator_scheduler=generator_scheduler,
#             scaler=scaler,
#             device=device,
#             gradient_accumulation_steps=gradient_accumulation_steps,
#             max_grad_norm=max_grad_norm,
#             epoch=epoch + 1
#         )
#
#         # Validation phase
#         val_metrics = validate_epoch(
#             image_encoder=image_encoder,
#             alignment_model=alignment_model,
#             report_generator=report_generator,
#             val_loader=val_loader,
#             clinical_bert_scorer=clinical_bert_scorer,
#             device=device,
#             epoch=epoch + 1
#         )
#
#         # Log metrics
#         if use_wandb:
#             wandb.log({**train_metrics, **val_metrics})
#
#         # Print training and validation losses
#         print(f"Epoch {epoch + 1} Training Loss: {train_metrics['train_loss']:.4f}")
#         print(f"Epoch {epoch + 1} Validation Loss: {val_metrics['val_loss']:.4f}")
#
#         # Save checkpoint if best model
#         current_val_score = val_metrics['val_bleu'] + val_metrics['val_rouge_l']
#         is_best = current_val_score > best_val_score
#         if is_best:
#             best_val_score = current_val_score
#
#         save_checkpoint({
#             'epoch': epoch + 1,
#             'alignment_model_state_dict': alignment_model.state_dict(),
#             'report_generator_state_dict': report_generator.state_dict(),
#             'generator_optimizer_state_dict': generator_optimizer.state_dict(),
#             'generator_scheduler_state_dict': generator_scheduler.state_dict(),
#             'best_val_score': best_val_score,
#             'train_metrics': train_metrics,
#             'val_metrics': val_metrics,
#         }, is_best, checkpoint_dir)
#
#         # Early stopping check
#         early_stopping(val_metrics['val_loss'])
#         if early_stopping.early_stop:
#             print("Early stopping triggered")
#             break
#
#     if use_wandb:
#         wandb.finish()
#
#
# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
#
#     # Direct path assignments
#     csv_with_image_paths = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/final.csv"
#     csv_with_labels = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/labeled_reports_with_images.csv"
#
#     # Training configuration
#     config = {
#         'num_epochs': 30,
#         'batch_size': 8,
#         'learning_rate': 1e-5,
#         'warmup_steps': 1000,
#         'gradient_accumulation_steps': 4,
#         'use_wandb': True,
#         'checkpoint_dir': 'checkpoints',
#         'seed': 42
#     }
#
#     # Start training
#     train_model(
#         csv_with_image_paths=csv_with_image_paths,
#         csv_with_labels=csv_with_labels,
#         **config
#     )

# train.py - corrected alignment
# train.py

# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from transformers import get_linear_schedule_with_warmup
# from tqdm import tqdm
# import logging
# import wandb
# from pathlib import Path
# from typing import Dict, Any
# from torch.cuda.amp import autocast, GradScaler
# from datetime import datetime
#
# from data2 import data_processing
# from alignment_model import ImageTextAlignmentModel
# from report_generator import MedicalReportGenerator
# from biovil_t.pretrained import get_biovil_t_image_encoder
# from rouge_score import rouge_scorer
#
# def train_epoch(image_encoder, alignment_model, report_generator, train_loader,
#                 contrastive_loss, alignment_optimizer, generator_optimizer,
#                 alignment_scheduler, generator_scheduler, scaler, device,
#                 gradient_accumulation_steps, max_grad_norm, epoch):
#     alignment_model.train()
#     report_generator.train()
#     image_encoder.eval()
#
#     # Metrics tracking
#     total_train_loss = 0.0
#     total_align_loss = 0.0
#     total_gen_loss = 0.0
#     total_samples = 0
#
#     progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
#
#     for batch_idx, (images, findings_texts, findings_lists) in enumerate(progress_bar):
#         images = images.to(device)
#         batch_size = images.size(0)
#         total_samples += batch_size
#
#         # Get image embeddings
#         with torch.no_grad():
#             image_embeddings = image_encoder(images).img_embedding
#
#         # Create prompts using findings_lists (for generation)
#         batch_prompts = [
#             f"Findings: {', '.join(findings) if findings else 'No Findings'}."
#             for findings in findings_lists
#         ]
#
#         # Use findings_texts (actual findings) for alignment
#         actual_findings = findings_texts
#
#         # Mixed precision training
#         with autocast():
#             # Alignment phase
#             projected_image, projected_text = alignment_model(image_embeddings, actual_findings)
#
#             # Contrastive loss
#             labels = torch.ones(batch_size).to(device)
#             align_loss = contrastive_loss(projected_image, projected_text, labels)
#             align_loss = align_loss / gradient_accumulation_steps
#
#         # Scale and accumulate alignment gradients
#         scaler.scale(align_loss).backward()
#
#         # Generation phase
#
#         # Tokenize the prompts
#         prompt_encoding = report_generator.tokenizer(
#             batch_prompts,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512
#         ).to(device)
#
#         # Tokenize target texts (actual findings)
#         target_encoding = report_generator.tokenizer(
#             actual_findings,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512
#         ).to(device)
#
#         with autocast():
#             gen_loss, _ = report_generator(
#                 image_embeddings=image_embeddings.detach(),
#                 prompt_input_ids=prompt_encoding['input_ids'],
#                 target_ids=target_encoding['input_ids']
#             )
#             gen_loss = gen_loss / gradient_accumulation_steps
#
#         # Scale and accumulate generator gradients
#         scaler.scale(gen_loss).backward()
#
#         # Update metrics
#         total_align_loss += align_loss.item() * gradient_accumulation_steps * batch_size
#         total_gen_loss += gen_loss.item() * gradient_accumulation_steps * batch_size
#         total_train_loss += (align_loss.item() + gen_loss.item()) * gradient_accumulation_steps * batch_size
#
#         # Step optimizers and schedulers
#         if (batch_idx + 1) % gradient_accumulation_steps == 0:
#             # Unscale gradients
#             scaler.unscale_(alignment_optimizer)
#             scaler.unscale_(generator_optimizer)
#
#             # Clip gradients
#             torch.nn.utils.clip_grad_norm_(
#                 alignment_model.parameters(), max_grad_norm
#             )
#             torch.nn.utils.clip_grad_norm_(
#                 report_generator.parameters(), max_grad_norm
#             )
#
#             # Step optimizers
#             scaler.step(alignment_optimizer)
#             scaler.step(generator_optimizer)
#             scaler.update()
#
#             # Zero gradients
#             alignment_optimizer.zero_grad()
#             generator_optimizer.zero_grad()
#
#             # Step schedulers
#             alignment_scheduler.step()
#             generator_scheduler.step()
#
#         # Update progress bar
#         progress_bar.set_postfix({
#             'align_loss': f"{align_loss.item():.4f}",
#             'gen_loss': f"{gen_loss.item():.4f}"
#         })
#
#     epoch_align_loss = total_align_loss / total_samples
#     epoch_gen_loss = total_gen_loss / total_samples
#     epoch_train_loss = total_train_loss / total_samples
#
#     return {
#         'train_loss': epoch_train_loss,
#         'train_align_loss': epoch_align_loss,
#         'train_gen_loss': epoch_gen_loss,
#     }
#
# def validate_epoch(image_encoder, alignment_model, report_generator, val_loader,
#                    contrastive_loss, device, epoch):
#     alignment_model.eval()
#     report_generator.eval()
#     image_encoder.eval()
#
#     # Metrics storage
#     total_val_loss = 0.0
#     total_align_loss = 0.0
#     total_gen_loss = 0.0
#     total_samples = 0
#     all_generated = []
#     all_references = []
#
#     with torch.no_grad():
#         progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
#
#         for batch_idx, (images, findings_texts, findings_lists) in enumerate(progress_bar):
#             images = images.to(device)
#             batch_size = images.size(0)
#             total_samples += batch_size
#
#             # Get image embeddings
#             image_embeddings = image_encoder(images).img_embedding
#
#             # Create prompts using findings_lists
#             batch_prompts = [
#                 f"Findings: {', '.join(findings) if findings else 'No Findings'}."
#                 for findings in findings_lists
#             ]
#
#             # Actual findings for alignment and reference
#             actual_findings = findings_texts
#
#             # Alignment phase
#             projected_image, projected_text = alignment_model(image_embeddings, actual_findings)
#             labels = torch.ones(batch_size).to(device)
#             align_loss = contrastive_loss(projected_image, projected_text, labels)
#
#             # Generation phase
#             prompt_encoding = report_generator.tokenizer(
#                 batch_prompts,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=512
#             ).to(device)
#
#             target_encoding = report_generator.tokenizer(
#                 actual_findings,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=512
#             ).to(device)
#
#             # Compute generation loss
#             gen_loss, _ = report_generator(
#                 image_embeddings=image_embeddings,
#                 prompt_input_ids=prompt_encoding['input_ids'],
#                 target_ids=target_encoding['input_ids']
#             )
#
#             # Generate text for evaluation
#             generated_texts = report_generator(
#                 image_embeddings=image_embeddings,
#                 prompt_input_ids=prompt_encoding['input_ids'],
#                 target_ids=None
#             )
#
#             # Store the generated and reference texts for ROUGE calculation
#             all_generated.extend(generated_texts)
#             all_references.extend(actual_findings)
#
#             # Update totals
#             total_align_loss += align_loss.item() * batch_size
#             total_gen_loss += gen_loss.item() * batch_size
#             total_val_loss += (align_loss.item() + gen_loss.item()) * batch_size
#
#             # Print sample generation
#             if batch_idx % 10 == 0:
#                 print(f"\nSample Generation (Batch {batch_idx}):")
#                 print(f"Generated: {generated_texts[0]}")
#                 print(f"Reference: {actual_findings[0]}")
#                 # Also display the pathologies findings from findings_lists
#                 print(f"Pathologies/Findings List: {findings_lists[0]}\n")
#
#         # Calculate overall metrics
#         epoch_align_loss = total_align_loss / total_samples
#         epoch_gen_loss = total_gen_loss / total_samples
#         epoch_val_loss = total_val_loss / total_samples
#
#     # Compute ROUGE-L
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     rouge_l_scores = []
#     for ref, gen in zip(all_references, all_generated):
#         score = scorer.score(ref, gen)['rougeL'].fmeasure
#         rouge_l_scores.append(score)
#     avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
#
#     # Display validation losses and ROUGE-L
#     print(f"\nEpoch {epoch} Validation Metrics:")
#     print(f"Validation Loss: {epoch_val_loss:.4f}")
#     print(f"Alignment Loss: {epoch_align_loss:.4f}")
#     print(f"Generation Loss: {epoch_gen_loss:.4f}")
#     print(f"ROUGE-L: {avg_rouge_l:.4f}")
#
#     return {
#         'val_loss': epoch_val_loss,
#         'val_align_loss': epoch_align_loss,
#         'val_gen_loss': epoch_gen_loss,
#         'val_rouge_l': avg_rouge_l
#     }
#
#
# def train_model(
#         csv_with_image_paths: str,
#         csv_with_labels: str,
#         num_epochs: int = 30,
#         batch_size: int = 8,
#         train_split: float = 0.85,
#         num_workers: int = 4,
#         learning_rate: float = 2e-5,
#         warmup_steps: int = 1000,
#         gradient_accumulation_steps: int = 4,
#         max_grad_norm: float = 1.0,
#         use_wandb: bool = True,
#         checkpoint_dir: str = "checkpoints",
#         seed: int = 42
# ):
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Initialize models
#     image_encoder = get_biovil_t_image_encoder()
#     alignment_model = ImageTextAlignmentModel(image_embedding_dim=512)
#     report_generator = MedicalReportGenerator(image_embedding_dim=512)
#
#     # Move models to device
#     image_encoder = image_encoder.to(device)
#     alignment_model = alignment_model.to(device)
#     report_generator = report_generator.to(device)
#
#     # Initialize wandb
#     if use_wandb:
#         wandb.init(
#             project="medical-report-generation",
#             config={
#                 "learning_rate": learning_rate,
#                 "epochs": num_epochs,
#                 "batch_size": batch_size,
#                 "warmup_steps": warmup_steps,
#                 "gradient_accumulation_steps": gradient_accumulation_steps,
#             }
#         )
#
#     # Get dataloaders
#     train_loader, val_loader = data_processing.get_dataloaders(
#         csv_with_image_paths=csv_with_image_paths,
#         csv_with_labels=csv_with_labels,
#         batch_size=batch_size,
#         train_split=train_split,
#         num_workers=num_workers,
#         seed=seed,
#     )
#
#     # Initialize optimizers
#     alignment_optimizer = AdamW(
#         alignment_model.parameters(),
#         lr=learning_rate,
#         weight_decay=0.01
#     )
#     generator_optimizer = AdamW([
#         {'params': report_generator.model.parameters(), 'lr': learning_rate},
#         {'params': report_generator.image_projection.parameters(), 'lr': learning_rate * 10}
#     ])
#
#     # Initialize schedulers
#     num_training_steps = len(train_loader) * num_epochs
#     alignment_scheduler = get_linear_schedule_with_warmup(
#         alignment_optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=num_training_steps
#     )
#     generator_scheduler = get_linear_schedule_with_warmup(
#         generator_optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=num_training_steps
#     )
#
#     # Initialize loss function and scaler
#     contrastive_loss = nn.CosineEmbeddingLoss()
#     scaler = GradScaler()
#
#     # Create checkpoint directory
#     checkpoint_dir = Path(checkpoint_dir)
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#
#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch + 1}/{num_epochs}")
#
#         # Training phase
#         train_metrics = train_epoch(
#             image_encoder=image_encoder,
#             alignment_model=alignment_model,
#             report_generator=report_generator,
#             train_loader=train_loader,
#             contrastive_loss=contrastive_loss,
#             alignment_optimizer=alignment_optimizer,
#             generator_optimizer=generator_optimizer,
#             alignment_scheduler=alignment_scheduler,
#             generator_scheduler=generator_scheduler,
#             scaler=scaler,
#             device=device,
#             gradient_accumulation_steps=gradient_accumulation_steps,
#             max_grad_norm=max_grad_norm,
#             epoch=epoch + 1
#         )
#
#         # Validation phase
#         val_metrics = validate_epoch(
#             image_encoder=image_encoder,
#             alignment_model=alignment_model,
#             report_generator=report_generator,
#             val_loader=val_loader,
#             contrastive_loss=contrastive_loss,
#             device=device,
#             epoch=epoch + 1
#         )
#
#         # Display training and validation losses
#         print(f"\nEpoch {epoch + 1} Training Loss: {train_metrics['train_loss']:.4f}")
#         print(f"Epoch {epoch + 1} Validation Loss: {val_metrics['val_loss']:.4f}")
#         print(f"Alignment Loss - Train: {train_metrics['train_align_loss']:.4f}, Val: {val_metrics['val_align_loss']:.4f}")
#         print(f"Generation Loss - Train: {train_metrics['train_gen_loss']:.4f}, Val: {val_metrics['val_gen_loss']:.4f}")
#         print(f"ROUGE-L (Val): {val_metrics['val_rouge_l']:.4f}")
#
#         # Log metrics to wandb
#         if use_wandb:
#             wandb.log({**train_metrics, **val_metrics})
#
#         # Save model checkpoints after each epoch
#         alignment_save_path = checkpoint_dir / f"alignment_model_epoch_{epoch+1}.pt"
#         report_generator_save_path = checkpoint_dir / f"report_generator_epoch_{epoch+1}.pt"
#
#         # Save alignment model
#         torch.save(alignment_model.state_dict(), alignment_save_path)
#         # Save report generator model
#         torch.save(report_generator.state_dict(), report_generator_save_path)
#
#         # Optionally, also save tokenizer (if needed)
#         # The tokenizer is from HuggingFace - you can save its vocab
#         tokenizer_save_path = checkpoint_dir / "tokenizer"
#         if not tokenizer_save_path.exists():
#             tokenizer_save_path.mkdir(parents=True, exist_ok=True)
#         report_generator.tokenizer.save_pretrained(str(tokenizer_save_path))
#
#     if use_wandb:
#         wandb.finish()
#
#
# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
#
#     # Path to your CSV files
#     csv_with_image_paths = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/final.csv"
#     csv_with_labels = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/labeled_reports_with_images.csv"
#
#     # Training configuration
#     config = {
#         'num_epochs': 30,
#         'batch_size': 8,
#         'learning_rate': 1e-5,
#         'warmup_steps': 1000,
#         'gradient_accumulation_steps': 4,
#         'use_wandb': True,
#         'checkpoint_dir': 'checkpoints',
#         'seed': 42
#     }
#
#     # Start training
#     train_model(
#         csv_with_image_paths=csv_with_image_paths,
#         csv_with_labels=csv_with_labels,
#         **config
#     )

# BioGPT with full data updated metrics

# # train.py - updated with metric calculations
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from pathlib import Path
import json
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Ensure NLTK packages are downloaded:
# nltk.download('wordnet')

from data2 import data_processing
from alignment_model import ImageTextAlignmentModel
from report_generator import MedicalReportGenerator
from biovil_t.model import ImageModel
from biovil_t.pretrained import get_biovil_t_image_encoder


def save_checkpoint(epoch: int, alignment_model: nn.Module, report_generator: nn.Module,
                   alignment_optimizer: torch.optim.Optimizer, generator_optimizer: torch.optim.Optimizer,
                   metrics: dict, save_path: Path) -> None:
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'alignment_model_state_dict': alignment_model.state_dict(),
        'report_generator_model': report_generator.model.state_dict(),
        'report_generator_projection': report_generator.input_projection.state_dict(),
        'alignment_optimizer_state_dict': alignment_optimizer.state_dict(),
        'generator_optimizer_state_dict': generator_optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)


def save_best_model(alignment_model: nn.Module, report_generator: nn.Module, metrics: dict, save_dir: Path) -> None:
    """Save best model with metrics"""
    # Save alignment model
    torch.save(alignment_model.state_dict(), save_dir / "best_alignment_model.pt")

    # Save report generator (PEFT model)
    report_generator.model.save_pretrained(save_dir / "best_report_generator")
    torch.save(report_generator.input_projection.state_dict(),
               save_dir / "best_report_generator_projection.pt")

    # Save metrics
    with open(save_dir / 'best_model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

def compute_metrics(references, predictions):
    """
    Compute ROUGE-L, BLEU-1, and METEOR scores.
    references: List of reference strings
    predictions: List of predicted strings
    """
    # Tokenize for BLEU and METEOR
    # Simple whitespace tokenization
    try:
        # Tokenize the references and predictions
        ref_tokens = [[ref.split()] for ref in references]  # corpus_bleu expects a list of lists of refs
        pred_tokens = [pred.split() for pred in predictions]

        # Compute BLEU-1 score
        # We use weights=(1,0,0,0) for BLEU-1
        bleu_1_score = corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0)) * 100.0

    except Exception as e:
        # print(f"An error occurred while calculating BLEU score")
        bleu_1_score = 0.0

    # Initialize the average METEOR score
    try:
        # METEOR score computation
        # meteor_score(ref, pred) expects strings, so we pass them directly
        meteor_scores = []
        for r, p in zip(references, predictions):
            try:
                score = meteor_score([r], p)
                meteor_scores.append(score)
            except Exception as e:
                # print(f"Error calculating METEOR for a pair")
                meteor_scores.append(0.0)  # Assign 0.0 to individual errors

        # Compute the average METEOR score
        avg_meteor = (sum(meteor_scores) / len(meteor_scores)) * 100.0 if meteor_scores else 0.0

    except Exception as e:
        # print(f"An error occurred while calculating the average METEOR score")
        avg_meteor = 0.0

    # Initialize the average ROUGE-L score
    try:
        # Initialize the ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l_scores = []

        # Compute ROUGE-L scores
        for r, p in zip(references, predictions):
            try:
                score = scorer.score(r, p)['rougeL'].fmeasure
                rouge_l_scores.append(score)
            except Exception as e:
                # print(f"Error calculating ROUGE-L for a pair")
                rouge_l_scores.append(0.0)  # Assign 0.0 to individual errors

        # Compute the average ROUGE-L score
        avg_rouge_l = (sum(rouge_l_scores) / len(rouge_l_scores)) * 100.0 if rouge_l_scores else 0.0

    except Exception as e:
        # print(f"An error occurred while calculating the average ROUGE-L score")
        avg_rouge_l = 0.0

    return avg_rouge_l, bleu_1_score, avg_meteor


def train_model(csv_path: str, save_dir: str, num_epochs: int = 30):
    """
    Train the medical report generation model

    Args:
        csv_path: Path to CSV file containing image paths and reports
        save_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models
    image_encoder = get_biovil_t_image_encoder()
    alignment_model = ImageTextAlignmentModel(image_embedding_dim=512)
    report_generator = MedicalReportGenerator()

    # Move models to device
    image_encoder = image_encoder.to(device)
    alignment_model = alignment_model.to(device)
    report_generator = report_generator.to(device)

    # Get dataloaders
    train_loader, val_loader = data_processing.get_dataloaders(csv_path)

    # Optimizers
    alignment_optimizer = AdamW(alignment_model.parameters(), lr=2e-5)
    peft_params = [p for p in report_generator.model.parameters() if p.requires_grad]
    generator_optimizer = AdamW([
        {'params': peft_params, 'lr': 2e-5},
        {'params': report_generator.input_projection.parameters(), 'lr': 1e-4}
    ])

    # Loss function for alignment
    contrastive_loss = nn.CosineEmbeddingLoss()

    # Create save directories
    save_dir = Path(save_dir)
    checkpoints_dir = save_dir / "checkpoints"
    best_model_dir = save_dir / "best_model"

    for dir_path in [checkpoints_dir, best_model_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)

    # Track best validation metrics
    best_val_loss = float('inf')
    best_metrics = None

    # Load last checkpoint if exists
    last_checkpoint = max(checkpoints_dir.glob("checkpoint_*.pt"), default=None,
                          key=lambda x: int(x.stem.split('_')[1]))
    start_epoch = 0

    if last_checkpoint:
        print(f"Loading checkpoint: {last_checkpoint}")
        checkpoint = torch.load(last_checkpoint, map_location=device)
        alignment_model.load_state_dict(checkpoint['alignment_model_state_dict'])
        report_generator.model.load_state_dict(checkpoint['report_generator_model'])
        report_generator.input_projection.load_state_dict(checkpoint['report_generator_projection'])
        alignment_optimizer.load_state_dict(checkpoint['alignment_optimizer_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training Phase
        image_encoder.eval()  # Keep image encoder in eval mode
        alignment_model.train()
        report_generator.train()

        # Track training losses
        train_align_losses = []
        train_gen_losses = []

        # Progress bar
        progress_bar = tqdm(train_loader, desc=f'Training')

        for batch_idx, (images, impressions) in enumerate(progress_bar):
            images = images.to(device)

            # Get image embeddings
            with torch.no_grad():
                image_embeddings = image_encoder(images).img_embedding

            # Alignment phase
            alignment_optimizer.zero_grad()
            projected_image, projected_text = alignment_model(image_embeddings, impressions)
            batch_size = images.size(0)
            labels = torch.ones(batch_size).to(device)
            align_loss = contrastive_loss(projected_image, projected_text, labels)
            align_loss.backward()
            alignment_optimizer.step()

            # Generation phase
            generator_optimizer.zero_grad()
            target_encoding = report_generator.tokenizer(
                impressions,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=150
            ).to(device)
            target_ids = target_encoding['input_ids']
            gen_loss, logits = report_generator(projected_image.detach(), target_ids)
            gen_loss.backward()
            generator_optimizer.step()

            # Track losses
            train_align_losses.append(align_loss.item())
            train_gen_losses.append(gen_loss.item())

            # Update progress bar
            progress_bar.set_postfix({
                'Align Loss': f'{align_loss.item():.4f}',
                'Gen Loss': f'{gen_loss.item():.4f}'
            })

            # Print sample outputs every 50 batches
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    sample_report = report_generator.generate_report(projected_image[0:1].detach())[0]
                    print("\nSample Generation:")
                    print(f"Generated: {sample_report}")
                    print(f"Target: {impressions[0]}\n")

        # Calculate average training losses
        avg_train_align_loss = sum(train_align_losses) / len(train_align_losses)
        avg_train_gen_loss = sum(train_gen_losses) / len(train_gen_losses)

        # Validation Phase
        alignment_model.eval()
        report_generator.eval()

        val_align_losses = []
        val_gen_losses = []
        val_references = []
        val_predictions = []

        print("\nRunning validation...")
        with torch.no_grad():
            for val_images, val_impressions in val_loader:
                val_images = val_images.to(device)
                val_image_embeddings = image_encoder(val_images).img_embedding
                val_projected_image, val_projected_text = alignment_model(val_image_embeddings, val_impressions)

                # Calculate alignment loss
                val_labels = torch.ones(val_images.size(0)).to(device)
                val_align_loss = contrastive_loss(val_projected_image, val_projected_text, val_labels)
                val_align_losses.append(val_align_loss.item())

                # Prepare target text for generation loss
                val_target_encoding = report_generator.tokenizer(
                    val_impressions,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=150
                ).to(device)
                val_target_ids = val_target_encoding['input_ids']

                val_gen_loss, _ = report_generator(val_projected_image, val_target_ids)
                val_gen_losses.append(val_gen_loss.item())

                # Generate predictions for metrics
                batch_predictions = report_generator.generate_report(val_projected_image)
                val_predictions.extend(batch_predictions)
                val_references.extend(val_impressions)

        # Calculate average validation losses
        avg_val_align_loss = sum(val_align_losses) / len(val_align_losses)
        avg_val_gen_loss = sum(val_gen_losses) / len(val_gen_losses)

        # Compute metrics
        rouge_l, bleu_1, meteor = compute_metrics(val_references, val_predictions)

        # Current metrics
        current_metrics = {
            'epoch': epoch + 1,
            'train_align_loss': avg_train_align_loss,
            'train_gen_loss': avg_train_gen_loss,
            'val_align_loss': avg_val_align_loss,
            'val_gen_loss': avg_val_gen_loss,
            'rouge_l': rouge_l,
            'bleu_1': bleu_1,
            'meteor': meteor
        }

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(json.dumps(current_metrics, indent=4))

        # Save checkpoint for each epoch
        checkpoint_path = checkpoints_dir / f"checkpoint_{epoch}.pt"
        save_checkpoint(
            epoch=epoch,
            alignment_model=alignment_model,
            report_generator=report_generator,
            alignment_optimizer=alignment_optimizer,
            generator_optimizer=generator_optimizer,
            metrics=current_metrics,
            save_path=checkpoint_path
        )
        print(f"\nSaved checkpoint to {checkpoint_path}")

        # Save best model if validation loss improved
        val_loss = avg_val_align_loss + avg_val_gen_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = current_metrics
            print("\nSaving best model...")
            save_best_model(alignment_model, report_generator, current_metrics, best_model_dir)

            # Save pointer to best checkpoint
            with open(save_dir / "best_checkpoint.txt", "w") as f:
                f.write(f"checkpoint_{epoch}.pt")

    print("\nTraining completed!")
    if best_metrics:
        print("\nBest model metrics:")
        print(json.dumps(best_metrics, indent=4))

    return alignment_model, report_generator


if __name__ == "__main__":
    # Set paths
    csv_path = "Data/new_csv_17k_rows.csv"  # Update this to your CSV path
    save_dir = "checkpoints"  # Directory where models will be saved

    # Call training function
    print("Starting training...")
    alignment_model, report_generator = train_model(csv_path=csv_path,
                                                    save_dir=save_dir,
                                                    num_epochs=30)
    print("Training completed!")

# train.py - CXRMate
# train.py
# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from tqdm import tqdm
# import logging
# from pathlib import Path
# import nltk
# from nltk.translate.bleu_score import corpus_bleu
# from nltk.translate.meteor_score import meteor_score
# from rouge_score import rouge_scorer
#
# from transformers import AutoTokenizer, AutoModel
# from data2 import data_processing
# from model_setup import setup_model_for_finetuning
#
# # nltk.download('wordnet')
#
# def compute_metrics(references, predictions):
#     try:
#         # BLEU-1 Calculation with Error Handling
#         try:
#             ref_tokens = [[ref.split()] for ref in references]
#             pred_tokens = [pred.split() for pred in predictions]
#             bleu_1_score = corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0)) * 100.0
#         except Exception as e:
#             # print(f"Error calculating BLEU-1 score: {e}")
#             bleu_1_score = 0.0
#
#         # METEOR Calculation with Error Handling
#         try:
#             meteor_scores = []
#             for r, p in zip(references, predictions):
#                 try:
#                     meteor_scores.append(meteor_score([r], p))
#                 except Exception as e:
#                     # print(f"Error calculating METEOR for a pair (ref: {r}, pred: {p}): {e}")
#                     meteor_scores.append(0.0)
#             avg_meteor = (sum(meteor_scores) / len(meteor_scores)) * 100.0 if meteor_scores else 0.0
#         except Exception as e:
#             # print(f"Error calculating average METEOR score: {e}")
#             avg_meteor = 0.0
#
#         # ROUGE-L Calculation with Error Handling
#         try:
#             scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#             rouge_l_scores = []
#             for r, p in zip(references, predictions):
#                 try:
#                     score = scorer.score(r, p)['rougeL'].fmeasure
#                     rouge_l_scores.append(score)
#                 except Exception as e:
#                     # print(f"Error calculating ROUGE-L for a pair (ref: {r}, pred: {p}): {e}")
#                     rouge_l_scores.append(0.0)
#             avg_rouge_l = (sum(rouge_l_scores) / len(rouge_l_scores)) * 100.0 if rouge_l_scores else 0.0
#         except Exception as e:
#             # print(f"Error calculating average ROUGE-L score: {e}")
#             avg_rouge_l = 0.0
#
#     except Exception as e:
#         # print(f"An unexpected error occurred: {e}")
#         bleu_1_score = avg_meteor = avg_rouge_l = 0.0
#
#     return avg_rouge_l, bleu_1_score, avg_meteor
#
# def train_model(csv_path: str, num_epochs: int = 20, batch_size: int = 8, lr=2e-5):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Starting Training.....")
#     print(f"Using device: {device}")
#
#     tokenizer = AutoTokenizer.from_pretrained("aehrc/cxrmate-single-tf", trust_remote_code=True)
#     model = AutoModel.from_pretrained("aehrc/cxrmate-single-tf", trust_remote_code=True)
#     model = model.to(device)
#
#     # Set up partial fine-tuning
#     model = setup_model_for_finetuning(model)
#
#     train_loader, val_loader = data_processing.get_dataloaders(csv_path, batch_size=batch_size)
#
#     optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
#
#     for epoch in range(num_epochs):
#         model.train()
#         train_losses = []
#         progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
#
#         for batch_idx, (images, findings) in enumerate(progress_bar):
#             # During training
#             images = images.to(device)
#             text_inputs = tokenizer(findings, padding=True, truncation=True, return_tensors="pt", max_length=150)
#             text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
#             decoder_input_ids = text_inputs['input_ids']  # For the decoder
#
#             # Forward pass without labels
#             outputs = model(pixel_values=images, input_ids=decoder_input_ids)
#             logits = outputs.logits
#             # Shift tokens for teacher forcing
#             shifted_labels = decoder_input_ids[:, 1:].clone()
#             shifted_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
#             shifted_labels = shifted_labels.reshape(-1)
#
#             loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
#             loss = loss_fn(shifted_logits, shifted_labels)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             train_losses.append(loss.item())
#             progress_bar.set_postfix({'Train Loss': f'{loss.item():.4f}'})
#
#             if batch_idx % 50 == 0:
#                 # Generate a sample
#                 model.eval()
#                 with torch.no_grad():
#                     gen_out = model.generate(pixel_values=images[0:1], max_length=50, num_return_sequences=1, do_sample=True, top_p=0.9, top_k=50)
#                 sample_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
#                 print("\nSample Generation:")
#                 print(f"Generated: {sample_text}")
#                 print(f"Target: {findings[0]}\n")
#                 model.train()
#
#         avg_train_loss = sum(train_losses) / len(train_losses)
#
#         # Validation
#         model.eval()
#         val_losses = []
#         val_references = []
#         val_predictions = []
#
#         with torch.no_grad():
#             for val_images, val_findings in val_loader:
#                 val_images = val_images.to(device)
#                 val_text_inputs = tokenizer(val_findings, padding=True, truncation=True, return_tensors="pt", max_length=150)
#                 val_text_inputs = {k: v.to(device) for k, v in val_text_inputs.items()}
#                 val_input_ids = val_text_inputs['input_ids']
#
#                 val_outputs = model(pixel_values=val_images, input_ids=val_input_ids, labels=val_input_ids)
#                 val_loss = val_outputs.loss
#                 val_losses.append(val_loss.item())
#
#                 # Generate predictions
#                 gen = model.generate(pixel_values=val_images, max_length=50, num_return_sequences=1, do_sample=False)
#                 batch_predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in gen]
#
#                 val_references.extend(val_findings)
#                 val_predictions.extend(batch_predictions)
#
#         avg_val_loss = sum(val_losses) / len(val_losses)
#         rouge_l, bleu_1, meteor = compute_metrics(val_references, val_predictions)
#
#         print(f"\nEpoch {epoch+1} Summary:")
#         print(f"Training Loss: {avg_train_loss:.4f}")
#         print(f"Validation Loss: {avg_val_loss:.4f}")
#         print(f"Validation ROUGE-L: {rouge_l:.2f}")
#         print(f"Validation BLEU-1: {bleu_1:.2f}")
#         print(f"Validation METEOR: {meteor:.2f}")
#
#         checkpoint_dir = Path("checkpoints")
#         checkpoint_dir.mkdir(exist_ok=True)
#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'train_loss': avg_train_loss,
#             'val_loss': avg_val_loss
#         }
#         torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
#
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     csv_path = "Data/final.csv"  # Update with your CSV path
#     train_model(csv_path)

# BioGPT correct model saving:

# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import logging
# from pathlib import Path
# import json
# import nltk
# from nltk.translate.bleu_score import corpus_bleu
# from nltk.translate.meteor_score import meteor_score
# from rouge_score import rouge_scorer
#
# # Ensure NLTK packages are downloaded if needed:
# # nltk.download('wordnet')
#
# from data2 import data_processing
# from alignment_model import ImageTextAlignmentModel
# from report_generator import MedicalReportGenerator
# from biovil_t.pretrained import get_biovil_t_image_encoder
#
#
# def save_checkpoint(epoch: int, alignment_model: nn.Module, report_generator: MedicalReportGenerator,
#                     alignment_optimizer: torch.optim.Optimizer, generator_optimizer: torch.optim.Optimizer,
#                     metrics: dict, save_path: Path) -> None:
#     """Save intermediate model checkpoint for training resumption"""
#     checkpoint = {
#         'epoch': epoch,
#         'alignment_model_state_dict': alignment_model.state_dict(),
#         'report_generator_model': report_generator.model.state_dict(),
#         'report_generator_projection': report_generator.input_projection.state_dict(),
#         'alignment_optimizer_state_dict': alignment_optimizer.state_dict(),
#         'generator_optimizer_state_dict': generator_optimizer.state_dict(),
#         'metrics': metrics
#     }
#     torch.save(checkpoint, save_path)
#
#
# def save_best_model(alignment_model: nn.Module, report_generator: MedicalReportGenerator, metrics: dict, save_dir: Path) -> None:
#     """Save the best model with metrics using the proper PEFT saving methods."""
#     # Save alignment model
#     torch.save(alignment_model.state_dict(), save_dir / "best_alignment_model.pt")
#
#     # Save report generator LoRA adapter weights and configuration
#     report_generator.model.save_pretrained(save_dir / "best_report_generator")
#
#     # Save the projection layer
#     torch.save(report_generator.input_projection.state_dict(),
#                save_dir / "best_report_generator_projection.pt")
#
#     # Save metrics
#     with open(save_dir / 'best_model_metrics.json', 'w') as f:
#         json.dump(metrics, f, indent=4)
#
#
# def compute_metrics(references, predictions):
#     """
#     Compute ROUGE-L, BLEU-1, and METEOR scores.
#     references: List of reference strings
#     predictions: List of predicted strings
#     """
#     try:
#         ref_tokens = [[ref.split()] for ref in references]
#         pred_tokens = [pred.split() for pred in predictions]
#         bleu_1_score = corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0)) * 100.0
#     except Exception:
#         bleu_1_score = 0.0
#
#     try:
#         meteor_scores = []
#         for r, p in zip(references, predictions):
#             try:
#                 score = meteor_score([r], p)
#                 meteor_scores.append(score)
#             except:
#                 meteor_scores.append(0.0)
#         avg_meteor = (sum(meteor_scores) / len(meteor_scores)) * 100.0 if meteor_scores else 0.0
#     except Exception:
#         avg_meteor = 0.0
#
#     try:
#         scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#         rouge_l_scores = []
#         for r, p in zip(references, predictions):
#             try:
#                 score = scorer.score(r, p)['rougeL'].fmeasure
#                 rouge_l_scores.append(score)
#             except:
#                 rouge_l_scores.append(0.0)
#         avg_rouge_l = (sum(rouge_l_scores) / len(rouge_l_scores)) * 100.0 if rouge_l_scores else 0.0
#     except Exception:
#         avg_rouge_l = 0.0
#
#     return avg_rouge_l, bleu_1_score, avg_meteor
#
#
# def train_model(csv_path: str, save_dir: str, num_epochs: int = 30):
#     """
#     Train the medical report generation model
#
#     Args:
#         csv_path: Path to CSV file containing image paths and reports
#         save_dir: Directory to save model checkpoints and final best model
#         num_epochs: Number of training epochs
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Initialize models
#     image_encoder = get_biovil_t_image_encoder()
#     alignment_model = ImageTextAlignmentModel(image_embedding_dim=512)
#     report_generator = MedicalReportGenerator()
#
#     # Move models to device
#     image_encoder = image_encoder.to(device)
#     alignment_model = alignment_model.to(device)
#     report_generator = report_generator.to(device)
#
#     # Get dataloaders
#     train_loader, val_loader = data_processing.get_dataloaders(csv_path)
#
#     # Optimizers
#     alignment_optimizer = AdamW(alignment_model.parameters(), lr=2e-5)
#     peft_params = [p for p in report_generator.model.parameters() if p.requires_grad]
#     generator_optimizer = AdamW([
#         {'params': peft_params, 'lr': 2e-5},
#         {'params': report_generator.input_projection.parameters(), 'lr': 1e-4}
#     ])
#
#     # Loss function for alignment
#     contrastive_loss = nn.CosineEmbeddingLoss()
#
#     # Create save directories
#     save_dir = Path(save_dir)
#     checkpoints_dir = save_dir / "checkpoints"
#     best_model_dir = save_dir / "best_model"
#
#     for dir_path in [checkpoints_dir, best_model_dir]:
#         dir_path.mkdir(exist_ok=True, parents=True)
#
#     # Track best validation metrics
#     best_val_loss = float('inf')
#     best_metrics = None
#
#     # Load last checkpoint if exists
#     last_checkpoint = max(checkpoints_dir.glob("checkpoint_*.pt"), default=None,
#                           key=lambda x: int(x.stem.split('_')[1]))
#     start_epoch = 0
#
#     if last_checkpoint:
#         print(f"Loading checkpoint: {last_checkpoint}")
#         checkpoint = torch.load(last_checkpoint, map_location=device)
#         alignment_model.load_state_dict(checkpoint['alignment_model_state_dict'])
#         report_generator.model.load_state_dict(checkpoint['report_generator_model'])
#         report_generator.input_projection.load_state_dict(checkpoint['report_generator_projection'])
#         alignment_optimizer.load_state_dict(checkpoint['alignment_optimizer_state_dict'])
#         generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
#         start_epoch = checkpoint['epoch'] + 1
#         print(f"Resuming from epoch {start_epoch}")
#
#     for epoch in range(start_epoch, num_epochs):
#         print(f"\nEpoch {epoch + 1}/{num_epochs}")
#
#         # Training Phase
#         image_encoder.eval()  # Keep image encoder in eval mode
#         alignment_model.train()
#         report_generator.train()
#
#         train_align_losses = []
#         train_gen_losses = []
#
#         progress_bar = tqdm(train_loader, desc='Training')
#
#         for batch_idx, (images, impressions) in enumerate(progress_bar):
#             images = images.to(device)
#
#             # Get image embeddings
#             with torch.no_grad():
#                 image_embeddings = image_encoder(images).img_embedding
#
#             # Alignment phase
#             alignment_optimizer.zero_grad()
#             projected_image, projected_text = alignment_model(image_embeddings, impressions)
#             batch_size = images.size(0)
#             labels = torch.ones(batch_size).to(device)
#             align_loss = contrastive_loss(projected_image, projected_text, labels)
#             align_loss.backward()
#             alignment_optimizer.step()
#
#             # Generation phase
#             generator_optimizer.zero_grad()
#             target_encoding = report_generator.tokenizer(
#                 impressions,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=150
#             ).to(device)
#             target_ids = target_encoding['input_ids']
#             gen_loss, logits = report_generator(projected_image.detach(), target_ids)
#             gen_loss.backward()
#             generator_optimizer.step()
#
#             train_align_losses.append(align_loss.item())
#             train_gen_losses.append(gen_loss.item())
#
#             progress_bar.set_postfix({
#                 'Align Loss': f'{align_loss.item():.4f}',
#                 'Gen Loss': f'{gen_loss.item():.4f}'
#             })
#
#             # Print sample outputs every 50 batches
#             if batch_idx % 50 == 0:
#                 with torch.no_grad():
#                     sample_report = report_generator.generate_report(projected_image[0:1].detach())[0]
#                     print("\nSample Generation:")
#                     print(f"Generated: {sample_report}")
#                     print(f"Target: {impressions[0]}\n")
#
#         # Calculate average training losses
#         avg_train_align_loss = sum(train_align_losses) / len(train_align_losses)
#         avg_train_gen_loss = sum(train_gen_losses) / len(train_gen_losses)
#
#         # Validation Phase
#         alignment_model.eval()
#         report_generator.eval()
#
#         val_align_losses = []
#         val_gen_losses = []
#         val_references = []
#         val_predictions = []
#
#         print("\nRunning validation...")
#         with torch.no_grad():
#             for val_images, val_impressions in val_loader:
#                 val_images = val_images.to(device)
#                 val_image_embeddings = image_encoder(val_images).img_embedding
#                 val_projected_image, val_projected_text = alignment_model(val_image_embeddings, val_impressions)
#
#                 # Alignment loss
#                 val_labels = torch.ones(val_images.size(0)).to(device)
#                 val_align_loss = contrastive_loss(val_projected_image, val_projected_text, val_labels)
#                 val_align_losses.append(val_align_loss.item())
#
#                 # Generation loss
#                 val_target_encoding = report_generator.tokenizer(
#                     val_impressions,
#                     padding=True,
#                     truncation=True,
#                     return_tensors="pt",
#                     max_length=150
#                 ).to(device)
#                 val_target_ids = val_target_encoding['input_ids']
#
#                 val_gen_loss, _ = report_generator(val_projected_image, val_target_ids)
#                 val_gen_losses.append(val_gen_loss.item())
#
#                 # Generate predictions for metrics
#                 batch_predictions = report_generator.generate_report(val_projected_image)
#                 val_predictions.extend(batch_predictions)
#                 val_references.extend(val_impressions)
#
#         # Calculate average validation losses
#         avg_val_align_loss = sum(val_align_losses) / len(val_align_losses)
#         avg_val_gen_loss = sum(val_gen_losses) / len(val_gen_losses)
#
#         # Compute metrics
#         rouge_l, bleu_1, meteor = compute_metrics(val_references, val_predictions)
#
#         # Current metrics
#         current_metrics = {
#             'epoch': epoch + 1,
#             'train_align_loss': avg_train_align_loss,
#             'train_gen_loss': avg_train_gen_loss,
#             'val_align_loss': avg_val_align_loss,
#             'val_gen_loss': avg_val_gen_loss,
#             'rouge_l': rouge_l,
#             'bleu_1': bleu_1,
#             'meteor': meteor
#         }
#
#         print(f"\nEpoch {epoch + 1} Summary:")
#         print(json.dumps(current_metrics, indent=4))
#
#         # Save checkpoint for each epoch
#         checkpoint_path = checkpoints_dir / f"checkpoint_{epoch}.pt"
#         save_checkpoint(
#             epoch=epoch,
#             alignment_model=alignment_model,
#             report_generator=report_generator,
#             alignment_optimizer=alignment_optimizer,
#             generator_optimizer=generator_optimizer,
#             metrics=current_metrics,
#             save_path=checkpoint_path
#         )
#         print(f"\nSaved checkpoint to {checkpoint_path}")
#
#         # Save best model if validation loss improved
#         val_loss = avg_val_align_loss + avg_val_gen_loss
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_metrics = current_metrics
#             print("\nSaving best model...")
#             save_best_model(alignment_model, report_generator, current_metrics, best_model_dir)
#
#             # Save pointer to best checkpoint
#             with open(save_dir / "best_checkpoint.txt", "w") as f:
#                 f.write(f"checkpoint_{epoch}.pt")
#
#     print("\nTraining completed!")
#     if best_metrics:
#         print("\nBest model metrics:")
#         print(json.dumps(best_metrics, indent=4))
#
#     return alignment_model, report_generator

#
# if __name__ == "__main__":
#     # Set paths
#     csv_path = "Data/final.csv"  # Update this to your CSV path
#     save_dir = "checkpoints"      # Directory where models will be saved
#
#     # Start training
#     print("Starting training...")
#     alignment_model, report_generator = train_model(csv_path=csv_path,
#                                                     save_dir=save_dir,
#                                                     num_epochs=30)
#     print("Training completed!")

# Concat-Trail2
# train.py

# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from transformers import get_linear_schedule_with_warmup
# from tqdm import tqdm
# import logging
# import wandb
# from pathlib import Path
# from typing import Dict, Any
# from torch.cuda.amp import autocast, GradScaler
# from datetime import datetime
#
# from data2 import data_processing
# from alignment_model import ImageTextAlignmentModel
# from report_generator import MedicalReportGenerator
# from biovil_t.pretrained import get_biovil_t_image_encoder  # Ensure this import path is correct
# from rouge_score import rouge_scorer
#
# def train_epoch(image_encoder, alignment_model, report_generator, train_loader,
#                 contrastive_loss, alignment_optimizer, generator_optimizer,
#                 alignment_scheduler, generator_scheduler, scaler, device,
#                 gradient_accumulation_steps, max_grad_norm, epoch):
#     alignment_model.train()
#     report_generator.train()
#     image_encoder.eval()
#
#     # Metrics tracking
#     total_train_loss = 0.0
#     total_align_loss = 0.0
#     total_gen_loss = 0.0
#     total_samples = 0
#
#     progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
#
#     for batch_idx, (images, findings_texts, findings_lists) in enumerate(progress_bar):
#         images = images.to(device)
#         batch_size = images.size(0)
#         total_samples += batch_size
#
#         # Get image embeddings
#         with torch.no_grad():
#             image_embeddings = image_encoder(images).img_embedding
#
#         # Create prompts using findings_lists (for generation)
#         batch_prompts = [
#             f"Findings: {', '.join(findings) if findings else 'No Findings'}."
#             for findings in findings_lists
#         ]
#
#         # Use findings_texts (actual findings) for alignment
#         actual_findings = findings_texts
#
#         # Mixed precision training
#         with autocast():
#             # Alignment phase
#             projected_image, projected_text = alignment_model(image_embeddings, actual_findings)
#
#             # Contrastive loss
#             labels = torch.ones(batch_size).to(device)
#             align_loss = contrastive_loss(projected_image, projected_text, labels)
#             align_loss = align_loss / gradient_accumulation_steps
#
#         # Scale and accumulate alignment gradients
#         scaler.scale(align_loss).backward()
#
#         # Generation phase
#
#         # Tokenize the prompts
#         prompt_encoding = report_generator.tokenizer(
#             batch_prompts,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512
#         ).to(device)
#
#         # Tokenize target texts (actual findings)
#         target_encoding = report_generator.tokenizer(
#             actual_findings,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512
#         ).to(device)
#
#         with autocast():
#             gen_loss, _ = report_generator(
#                 image_embeddings=image_embeddings.detach(),
#                 prompt_input_ids=prompt_encoding['input_ids'],
#                 target_ids=target_encoding['input_ids']
#             )
#             gen_loss = gen_loss / gradient_accumulation_steps
#
#         # Scale and accumulate generator gradients
#         scaler.scale(gen_loss).backward()
#
#         # Update metrics
#         total_align_loss += align_loss.item() * gradient_accumulation_steps * batch_size
#         total_gen_loss += gen_loss.item() * gradient_accumulation_steps * batch_size
#         total_train_loss += (align_loss.item() + gen_loss.item()) * gradient_accumulation_steps * batch_size
#
#         # Step optimizers and schedulers
#         if (batch_idx + 1) % gradient_accumulation_steps == 0:
#             # Unscale gradients
#             scaler.unscale_(alignment_optimizer)
#             scaler.unscale_(generator_optimizer)
#
#             # Clip gradients
#             torch.nn.utils.clip_grad_norm_(
#                 alignment_model.parameters(), max_grad_norm
#             )
#             torch.nn.utils.clip_grad_norm_(
#                 report_generator.parameters(), max_grad_norm
#             )
#
#             # Step optimizers
#             scaler.step(alignment_optimizer)
#             scaler.step(generator_optimizer)
#             scaler.update()
#
#             # Zero gradients
#             alignment_optimizer.zero_grad()
#             generator_optimizer.zero_grad()
#
#             # Step schedulers
#             alignment_scheduler.step()
#             generator_scheduler.step()
#
#         # Update progress bar
#         progress_bar.set_postfix({
#             'align_loss': f"{align_loss.item():.4f}",
#             'gen_loss': f"{gen_loss.item():.4f}"
#         })
#
#     epoch_align_loss = total_align_loss / total_samples
#     epoch_gen_loss = total_gen_loss / total_samples
#     epoch_train_loss = total_train_loss / total_samples
#
#     return {
#         'train_loss': epoch_train_loss,
#         'train_align_loss': epoch_align_loss,
#         'train_gen_loss': epoch_gen_loss,
#     }
#
# def validate_epoch(image_encoder, alignment_model, report_generator, val_loader,
#                    contrastive_loss, device, epoch):
#     alignment_model.eval()
#     report_generator.eval()
#     image_encoder.eval()
#
#     # Metrics storage
#     total_val_loss = 0.0
#     total_align_loss = 0.0
#     total_gen_loss = 0.0
#     total_samples = 0
#     all_generated = []
#     all_references = []
#
#     with torch.no_grad():
#         progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
#
#         for batch_idx, (images, findings_texts, findings_lists) in enumerate(progress_bar):
#             images = images.to(device)
#             batch_size = images.size(0)
#             total_samples += batch_size
#
#             # Get image embeddings
#             image_embeddings = image_encoder(images).img_embedding
#
#             # Create prompts using findings_lists
#             batch_prompts = [
#                 f"Findings: {', '.join(findings) if findings else 'No Findings'}."
#                 for findings in findings_lists
#             ]
#
#             # Actual findings for alignment and reference
#             actual_findings = findings_texts
#
#             # Alignment phase
#             projected_image, projected_text = alignment_model(image_embeddings, actual_findings)
#             labels = torch.ones(batch_size).to(device)
#             align_loss = contrastive_loss(projected_image, projected_text, labels)
#
#             # Generation phase
#             prompt_encoding = report_generator.tokenizer(
#                 batch_prompts,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=512
#             ).to(device)
#
#             target_encoding = report_generator.tokenizer(
#                 actual_findings,
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#                 max_length=512
#             ).to(device)
#
#             # Compute generation loss
#             gen_loss, _ = report_generator(
#                 image_embeddings=image_embeddings,
#                 prompt_input_ids=prompt_encoding['input_ids'],
#                 target_ids=target_encoding['input_ids']
#             )
#
#             # Generate text for evaluation
#             generated_texts = report_generator(
#                 image_embeddings=image_embeddings,
#                 prompt_input_ids=prompt_encoding['input_ids'],
#                 target_ids=None
#             )
#
#             # Store the generated and reference texts for ROUGE calculation
#             all_generated.extend(generated_texts)
#             all_references.extend(actual_findings)
#
#             # Update totals
#             total_align_loss += align_loss.item() * batch_size
#             total_gen_loss += gen_loss.item() * batch_size
#             total_val_loss += (align_loss.item() + gen_loss.item()) * batch_size
#
#             # Print sample generation
#             if batch_idx % 10 == 0:
#                 print(f"\nSample Generation (Batch {batch_idx}):")
#                 print(f"Generated: {generated_texts[0]}")
#                 print(f"Reference: {actual_findings[0]}")
#                 # Also display the pathologies findings from findings_lists
#                 print(f"Pathologies/Findings List: {findings_lists[0]}\n")
#
#         # Calculate overall metrics
#         epoch_align_loss = total_align_loss / total_samples
#         epoch_gen_loss = total_gen_loss / total_samples
#         epoch_val_loss = total_val_loss / total_samples
#
#     # Compute ROUGE-L
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     rouge_l_scores = []
#     for ref, gen in zip(all_references, all_generated):
#         score = scorer.score(ref, gen)['rougeL'].fmeasure
#         rouge_l_scores.append(score)
#     avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
#
#     # Display validation losses and ROUGE-L
#     print(f"\nEpoch {epoch} Validation Metrics:")
#     print(f"Validation Loss: {epoch_val_loss:.4f}")
#     print(f"Alignment Loss: {epoch_align_loss:.4f}")
#     print(f"Generation Loss: {epoch_gen_loss:.4f}")
#     print(f"ROUGE-L: {avg_rouge_l:.4f}")
#
#     return {
#         'val_loss': epoch_val_loss,
#         'val_align_loss': epoch_align_loss,
#         'val_gen_loss': epoch_gen_loss,
#         'val_rouge_l': avg_rouge_l
#     }
#
#
# def train_model(
#         csv_with_image_paths: str,
#         csv_with_labels: str,
#         num_epochs: int = 30,
#         batch_size: int = 8,
#         train_split: float = 0.85,
#         num_workers: int = 4,
#         learning_rate: float = 2e-4,
#         warmup_steps: int = 1000,
#         gradient_accumulation_steps: int = 4,
#         max_grad_norm: float = 1.0,
#         use_wandb: bool = True,
#         checkpoint_dir: str = "checkpoints",
#         seed: int = 42
# ):
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Initialize models
#     image_encoder = get_biovil_t_image_encoder()
#     alignment_model = ImageTextAlignmentModel(image_embedding_dim=512)
#     report_generator = MedicalReportGenerator(image_embedding_dim=512)
#
#     # Move models to device
#     image_encoder = image_encoder.to(device)
#     alignment_model = alignment_model.to(device)
#     report_generator = report_generator.to(device)
#
#     # Initialize wandb
#     if use_wandb:
#         wandb.init(
#             project="medical-report-generation",
#             config={
#                 "learning_rate": learning_rate,
#                 "epochs": num_epochs,
#                 "batch_size": batch_size,
#                 "warmup_steps": warmup_steps,
#                 "gradient_accumulation_steps": gradient_accumulation_steps,
#             }
#         )
#         wandb.watch(models=[alignment_model, report_generator], log="all")
#
#     # Get dataloaders
#     train_loader, val_loader = data_processing.get_dataloaders(
#         csv_with_image_paths=csv_with_image_paths,
#         csv_with_labels=csv_with_labels,
#         batch_size=batch_size,
#         train_split=train_split,
#         num_workers=num_workers,
#         seed=seed,
#     )
#
#     # Initialize optimizers
#     alignment_optimizer = AdamW(
#         alignment_model.parameters(),
#         lr=learning_rate,
#         weight_decay=0.01
#     )
#     generator_optimizer = AdamW([
#         {'params': report_generator.model.parameters(), 'lr': learning_rate},
#         {'params': report_generator.image_projection.parameters(), 'lr': learning_rate * 10}
#     ])
#
#     # Initialize schedulers
#     num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
#     alignment_scheduler = get_linear_schedule_with_warmup(
#         alignment_optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=num_training_steps
#     )
#     generator_scheduler = get_linear_schedule_with_warmup(
#         generator_optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=num_training_steps
#     )
#
#     # Initialize loss function and scaler
#     contrastive_loss = nn.CosineEmbeddingLoss()
#     scaler = GradScaler()
#
#     # Create checkpoint directory
#     checkpoint_dir = Path(checkpoint_dir)
#     checkpoint_dir.mkdir(parents=True, exist_ok=True)
#
#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch + 1}/{num_epochs}")
#
#         # Training phase
#         train_metrics = train_epoch(
#             image_encoder=image_encoder,
#             alignment_model=alignment_model,
#             report_generator=report_generator,
#             train_loader=train_loader,
#             contrastive_loss=contrastive_loss,
#             alignment_optimizer=alignment_optimizer,
#             generator_optimizer=generator_optimizer,
#             alignment_scheduler=alignment_scheduler,
#             generator_scheduler=generator_scheduler,
#             scaler=scaler,
#             device=device,
#             gradient_accumulation_steps=gradient_accumulation_steps,
#             max_grad_norm=max_grad_norm,
#             epoch=epoch + 1
#         )
#
#         # Validation phase
#         val_metrics = validate_epoch(
#             image_encoder=image_encoder,
#             alignment_model=alignment_model,
#             report_generator=report_generator,
#             val_loader=val_loader,
#             contrastive_loss=contrastive_loss,
#             device=device,
#             epoch=epoch + 1
#         )
#
#         # Display training and validation losses
#         print(f"\nEpoch {epoch + 1} Training Loss: {train_metrics['train_loss']:.4f}")
#         print(f"Epoch {epoch + 1} Validation Loss: {val_metrics['val_loss']:.4f}")
#         print(f"Alignment Loss - Train: {train_metrics['train_align_loss']:.4f}, Val: {val_metrics['val_align_loss']:.4f}")
#         print(f"Generation Loss - Train: {train_metrics['train_gen_loss']:.4f}, Val: {val_metrics['val_gen_loss']:.4f}")
#         print(f"ROUGE-L (Val): {val_metrics['val_rouge_l']:.4f}")
#
#         # Log metrics to wandb
#         if use_wandb:
#             wandb.log({**train_metrics, **val_metrics})
#
#         # Save model checkpoint after each epoch
#         checkpoint_save_path = checkpoint_dir / f"model_epoch_{epoch+1}.pt"
#         torch.save({
#             'epoch': epoch + 1,
#             'image_encoder_state_dict': image_encoder.state_dict(),
#             'alignment_model_state_dict': alignment_model.state_dict(),
#             'report_generator_state_dict': report_generator.state_dict(),
#             'alignment_optimizer_state_dict': alignment_optimizer.state_dict(),
#             'generator_optimizer_state_dict': generator_optimizer.state_dict(),
#             'alignment_scheduler_state_dict': alignment_scheduler.state_dict(),
#             'generator_scheduler_state_dict': generator_scheduler.state_dict(),
#             'scaler_state_dict': scaler.state_dict(),
#             'config': {
#                 'learning_rate': learning_rate,
#                 'batch_size': batch_size,
#                 'gradient_accumulation_steps': gradient_accumulation_steps,
#                 'max_grad_norm': max_grad_norm,
#             }
#         }, checkpoint_save_path)
#         logging.info(f"Saved checkpoint: {checkpoint_save_path}")
#
#     if use_wandb:
#         wandb.finish()
#
#
# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
#
#     # Path to your CSV files
#     csv_with_image_paths = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/final.csv"
#     csv_with_labels = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/labeled_reports_with_images.csv"
#
#     # Training configuration
#     config = {
#         'num_epochs': 30,
#         'batch_size': 8,
#         'learning_rate': 1e-4,
#         'warmup_steps': 1000,
#         'gradient_accumulation_steps': 4,
#         'use_wandb': True,
#         'checkpoint_dir': 'checkpoints',
#         'seed': 42
#     }
#
#     # Start training
#     train_model(
#         csv_with_image_paths=csv_with_image_paths,
#         csv_with_labels=csv_with_labels,
#         **config
#     )
