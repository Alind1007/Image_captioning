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

