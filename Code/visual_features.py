import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import paths
import pandas as pd



class ScanVisualFeatures(nn.Module):
    def __init__(self):
        super(ScanVisualFeatures, self).__init__()
        self.model, self.features, self.avg_pool = self.get_model()

    def get_model(self):
        densenet = models.densenet121(weights='DEFAULT')
        layers = list(densenet.features)
        # 2nd last layer
        features = densenet.classifier.in_features

        # Create a sequential model with the extracted convolutional layers
        model = nn.Sequential(*layers)
        # Apply average pooling
        avg_pool = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        return model, features, avg_pool

    def forward(self, image):
        # Get features from last convolutional layer
        features = self.model(image)
        # Apply average pooling
        out_features = self.avg_pool(features).squeeze()

        return out_features


def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ) #image params
    ])


    image = Image.open(image_path).convert('RGB')


    image = transform(image)

    # adding a batch dim for processing (for ex shape would be (1,3,224,224))
    image = image.unsqueeze(0)

    return image


#testing

csv_path = paths.data_dir_path + 'image_report.csv' #path to dataset file which we created using data_preprocess

df1 = pd.read_csv(csv_path)

image_list = list(df1.image_path[:5].values)

feature_extractor = ScanVisualFeatures()

# Switch model to evaluation mode
feature_extractor.eval()

# Loop over the image paths
for image_path in image_list:
    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)

    # Ensure the image is in the correct device (CPU or GPU)
    image = image.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    feature_extractor = feature_extractor.to(image.device)

    # Extract features
    with torch.no_grad():  # Disable gradient calculation during inference
        features = feature_extractor(image)


    print(f"Extracted Features for {image_path}: {features.shape}")

    print(f"Feature Vector Mean: {features.mean().item()}")
    print(f"Feature Vector Standard Deviation: {features.std().item()}")








