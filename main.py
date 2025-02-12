import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import requests
from io import BytesIO
from PIL import Image
from google.colab import drive
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer

# Mount Google Drive (for accessing Drive images)
drive.mount('/content/drive')

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP model for feature extraction
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
blip_model.eval()

# Load GPT-2 for caption refinement
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

# Function to download and save an image from a URL
def download_image(image_url):
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # Check if request was successful
        img = Image.open(BytesIO(response.content))
        
        # Save image locally
        image_path = "downloaded_image.jpg"
        img.save(image_path)

        print(f"✅ Image downloaded successfully: {image_path}")
        return image_path  # Return local path
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Could not download image! {e}")
        return None


# Function to generate captions using BLIP
def generate_blip_caption(image_path):
    if not os.path.exists(image_path):
        print("❌ Error: Image file not found!")
        return None
    
    # Load and display image
    image = Image.open(image_path).convert("RGB")
    plt.imshow(image)
    plt.axis("off")
    plt.show()
