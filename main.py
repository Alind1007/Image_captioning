import os
import fitz  # PyMuPDF for text and image extraction
import re
import io
from PIL import Image
from playsound import playsound
import pytesseract
from equation_to_text import MathToSpeech  # Import MathToSpeech class


import torch
import soundfile as sf
import matplotlib.pyplot as plt
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# Load the processor, model, and vocoder from Hugging Face
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("Atrishi/speecht5_tts_voxpopuli_nl")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Generate a default speaker embedding (Neutral speaker)
speaker_embeddings = torch.zeros((1, 512))  # Assuming the model expects (1, 512) shape



def process_text(text, output_file="output.wav"):
    """
    Converts input text to speech and saves it as a WAV file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Process text input
    #model=model.to(device)
    inputs = processor(text=text, return_tensors="pt").to(device)

    # Generate spectrogram
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

    # Visualize spectrogram
    plt.figure()
    plt.imshow(spectrogram.cpu().T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.title("Generated Speech Spectrogram")
    plt.show()

    # Convert spectrogram to waveform
    with torch.no_grad():
        speech = vocoder(spectrogram)

    # Save audio file
    sf.write(output_file, speech.cpu().numpy(), samplerate=16000)
    print(f"Speech saved as {output_file}")

    return output_file  # Return filename


# Create an instance of MathToSpeech
math_to_speech = MathToSpeech()

#C:\Users\atris\Downloads\math_equations.pdf
#C:\Users\atris\Downloads\2025CSN362_L6 1.pdf
def process_document(file_path):
    """
    Processes a document, extracts text, handles images and equations, and maintains structure.
    """
    extracted_text = []
    
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Error opening file: {e}")
        return None

    for page_num, page in enumerate(doc):
        # Get the text blocks (this includes text along with their coordinates)
        text_blocks = page.get_text("dict")["blocks"]
        
        # Get the images from the page (with coordinates)
        images = page.get_images(full=True)

        # Merge text and images by their position on the page
        elements = []

        # Extract and sort text elements
        for block in text_blocks:
            #print("DEBUG BLOCK:", block)  # Debugging
            
            if "lines" in block:  # Check if text lines exist
                text_content = "\n".join(
                    span["text"] for line in block["lines"] if "spans" in line for span in line["spans"]
                )
                elements.append({"type": "text", "content": text_content, "y_pos": block["bbox"][1]})

        # Extract and sort image elements
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            # We use the Y-coordinate of the image for sorting
            img_y_pos = img[1]  # Image y-coordinate
            elements.append({"type": "image", "image": image, "y_pos": img_y_pos})

        # Sort elements by their Y-coordinate (preserve document order)
        elements.sort(key=lambda e: e["y_pos"])

        # Process text and images in order
        for element in elements:
            if element["type"] == "text":
                text = element["content"]
                
                # Detect and replace LaTeX equations
                latex_equations = re.findall(r"\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]", text)
                for eq in latex_equations:
                    text = text.replace(eq, f"[Equation]: {math_to_speech.process_latex_equation(eq)}")
                
                # Detect and replace text-based mathematical equations
                text_equations = re.findall(r"[A-Za-z0-9\s\+\-\*/\^=<>,;:!@#\$%&\(\)\{\}\[\]_]+", text)
                text_equations = [eq.strip() for eq in text_equations if eq.strip()]
                for eq in text_equations:
                    text = text.replace(eq, f"[Equation]: {math_to_speech.process_text_equation(eq)}")
            
                extracted_text.append(text)

            elif element["type"] == "image":
                # Process image (convert equation or caption)
                extracted_text.append(math_to_speech.process_image(element["image"]))

    # Combine all extracted text and generate speech
    final_text = '\n'.join(extracted_text)
    
    print(final_text)
    if final_text.strip():
        return process_text(final_text)  # Convert extracted text to speech
    else:
        print("No valid text extracted from the document.")
        return None



def process_input(input_data, input_type):
    """
    Processes user input: direct text or a document.
    """
    if input_type == 'text':
        return process_text(input_data)
    elif input_type == 'document':
        return process_document(input_data)
    else:
        raise ValueError("Invalid input type. Use 'text' or 'document'.")

import pygame


if __name__ == "__main__":
    input_type = input("Enter input type (text/document): ").strip().lower()
    
    if input_type == "text":
        text_input = input("Enter text: ")
        audio_output = process_input(text_input, 'text')  # Generate speech
        
        # Check if the file exists before playing
        if os.path.exists(audio_output):
            pygame.mixer.init()
            pygame.mixer.music.load(audio_output)
            pygame.mixer.music.play()
            
            print("Playing audio...")  # Indicate audio is playing
            
            while pygame.mixer.music.get_busy():  # Wait for playback to finish
                pygame.time.Clock().tick(10)
            
            print("Playback finished.")
        else:
            print(f"Audio file {audio_output} not found.")
        
    elif input_type == "document":
        file_path = input("Enter document file path: ").strip()
        audio_output = process_input(file_path, 'document')
        # Play the generated audio
        
        # Check if the file exists before playing
        if os.path.exists(audio_output):
            pygame.mixer.init()
            pygame.mixer.music.load(audio_output)
            pygame.mixer.music.play()
            
            print("Playing audio...")  # Indicate audio is playing
            
            while pygame.mixer.music.get_busy():  # Wait for playback to finish
                pygame.time.Clock().tick(10)
            
            print("Playback finished.")
        else:
            print(f"Audio file {audio_output} not found.")
        
    else:
        print("Invalid input type. Please enter 'text' or 'document'.")
    
    print("Speech generation completed.")
