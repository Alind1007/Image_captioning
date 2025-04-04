import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import requests
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer

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
    
    # Extract features and generate caption using BLIP
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        blip_features = blip_model.generate(**inputs)
    
    # Decode BLIP output
    blip_caption = processor.decode(blip_features[0], skip_special_tokens=True)
    
    print("\n📌 *BLIP Initial Caption:*", blip_caption)
    return blip_caption

# Function to refine the caption using GPT-2
def refine_caption(blip_caption):
    if blip_caption is None:
        return "❌ Error: No caption available to refine!"

    input_text = "Refine this caption: " + blip_caption
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = gpt2_model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)

    refined_caption = tokenizer.decode(output[0], skip_special_tokens=True).replace("Refine this caption:", "").strip()

    print("\n📌 *Final Refined Caption (GPT-2):*", refined_caption)
    return refined_caption

def convert_image_to_text(image):
    """
    Extracts text or generates captions from an image.
    """
    blip_caption = generate_blip_caption(image)
    refined_caption = refine_caption(blip_caption)
    return refined_caption

