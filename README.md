# Problem statement
The manual generation of chest X-ray reports by radiologists is time-consuming and prone to delays due to high workload, creating a need for an automated system that can produce accurate preliminary reports to support radiologists in their diagnostic workflow.


# Dataset: MIMIC-CXR

* 15,000 chest X-ray images (DICOM format)
* Associated radiology reports (XML format)
* 14 pathology labels including conditions like Pneumonia, Cardiomegaly, etc.

# Data Processing Pipeline

## Image Processing

* Download Data(code/download_folder.sh)
* DICOM to PNG conversion
* PNG Compression(code/CheXbert/src/split_data/compress_png.py)
* Created structured CSV with image metadata
* Applied data augmentation techniques


## Label Generation

* Utilized ChexBert for extracting labels from reports
* Dataset split: 85% training, 15% validation

# Model Architecture Exploration

## ChexNet Implementation

* Multi-label classification model
* Based on DenseNet-121 architecture
* Code can be found in (code/CheXbert/src/image_classifier/chexnet_train_class_imbal_2.py)

## ChexBert Implementation

## Tested Architectures

* BioVilt + Alignment + BioGPT. Code()
* BioVilt + ChexNet + Alignment + BioGPT
* BioMed + ChexNet + Alignment + BioGPT(code/biomed_blip/biomed/train_biomed_gpt_3.py)
* BLIP2 + ChexNet + BioGPT(code/biomed_blip/blip/train_no_blip_rouge.py)
* CXRMATE + ChexNet + BioGPT



# Key Findings

* ChexNet label integration improved overall performance
* BioVilt + ChexNet + BioGPT matched performance of larger BLIP model
* ROUGE-L metric used for evaluating report quality

# Technical Optimizations

* Implemented Parameter-Efficient Fine-Tuning (PEFT)
* Used LoRA for efficient model adaptation
* Combined image features with structural findings



