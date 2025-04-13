# import os
# import fitz  # PyMuPDF for text and image extraction
# import re
# import io
# from PIL import Image
# from playsound import playsound
# import pytesseract
# from equation_to_text import MathToSpeech  # Import MathToSpeech class


# import torch
# import soundfile as sf
# import matplotlib.pyplot as plt
# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# # Load the processor, model, and vocoder from Hugging Face
# processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# model = SpeechT5ForTextToSpeech.from_pretrained("Atrishi/speecht5_tts_voxpopuli_nl")
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# # Generate a default speaker embedding (Neutral speaker)
# speaker_embeddings = torch.zeros((1, 512))  # Assuming the model expects (1, 512) shape



# def process_text(text, output_file="output.wav"):
#     """
#     Converts input text to speech and saves it as a WAV file.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Process text input
#     #model=model.to(device)
#     inputs = processor(text=text, return_tensors="pt").to(device)

#     # Generate spectrogram
#     spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

#     # Visualize spectrogram
#     plt.figure()
#     plt.imshow(spectrogram.cpu().T, aspect="auto", origin="lower")
#     plt.colorbar()
#     plt.title("Generated Speech Spectrogram")
#     plt.show()

#     # Convert spectrogram to waveform
#     with torch.no_grad():
#         speech = vocoder(spectrogram)

#     # Save audio file
#     sf.write(output_file, speech.cpu().numpy(), samplerate=16000)
#     print(f"Speech saved as {output_file}")

#     return output_file  # Return filename


# # Create an instance of MathToSpeech
# math_to_speech = MathToSpeech()

# #C:\Users\atris\Downloads\math_equations.pdf
# #C:\Users\atris\Downloads\2025CSN362_L6 1.pdf
# def process_document(file_path):
#     """
#     Processes a document, extracts text, handles images and equations, and maintains structure.
#     """
#     extracted_text = []
    
#     try:
#         doc = fitz.open(file_path)
#     except Exception as e:
#         print(f"Error opening file: {e}")
#         return None

#     for page_num, page in enumerate(doc):
#         # Get the text blocks (this includes text along with their coordinates)
#         text_blocks = page.get_text("dict")["blocks"]
        
#         # Get the images from the page (with coordinates)
#         images = page.get_images(full=True)

#         # Merge text and images by their position on the page
#         elements = []

#         # Extract and sort text elements
#         for block in text_blocks:
#             #print("DEBUG BLOCK:", block)  # Debugging
            
#             if "lines" in block:  # Check if text lines exist
#                 text_content = "\n".join(
#                     span["text"] for line in block["lines"] if "spans" in line for span in line["spans"]
#                 )
#                 elements.append({"type": "text", "content": text_content, "y_pos": block["bbox"][1]})

#         # Extract and sort image elements
#         for img_index, img in enumerate(images):
#             xref = img[0]
#             base_image = doc.extract_image(xref)
#             image_bytes = base_image["image"]
#             image = Image.open(io.BytesIO(image_bytes))

#             # We use the Y-coordinate of the image for sorting
#             img_y_pos = img[1]  # Image y-coordinate
#             elements.append({"type": "image", "image": image, "y_pos": img_y_pos})

#         # Sort elements by their Y-coordinate (preserve document order)
#         elements.sort(key=lambda e: e["y_pos"])

#         # Process text and images in order
#         for element in elements:
#             if element["type"] == "text":
#                 text = element["content"]
                
#                 # Detect and replace LaTeX equations
#                 latex_equations = re.findall(r"\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]", text)
#                 for eq in latex_equations:
#                     text = text.replace(eq, f" {math_to_speech.process_latex_equation(eq)}")
                
#                 # Detect and replace text-based mathematical equations
#                 text_equations = re.findall(r"[A-Za-z0-9\s\+\-\*/\^=<>,;:!@#\$%&\(\)\{\}\[\]_]+", text)
#                 text_equations = [eq.strip() for eq in text_equations if eq.strip()]
#                 for eq in text_equations:
#                     text = text.replace(eq, f" {math_to_speech.process_text_equation(eq)}")
            
#                 extracted_text.append(text)

#             elif element["type"] == "image":
#                 # Process image (convert equation or caption)
#                 extracted_text.append(math_to_speech.process_image(element["image"]))


#     # Combine all extracted text and generate speech
#     final_text = '\n'.join(extracted_text)
    
#     #convert numbers to text
#     final_text=math_to_speech.process_numbers(final_text)
    
#     print(final_text)
#     if final_text.strip():
#         return process_text(final_text)  # Convert extracted text to speech
#     else:
#         print("No valid text extracted from the document.")
#         return None



# def process_input(input_data, input_type):
#     """
#     Processes user input: direct text or a document.
#     """
#     if input_type == 'text':
#         return process_text(input_data)
#     elif input_type == 'document':
#         return process_document(input_data)
#     else:
#         raise ValueError("Invalid input type. Use 'text' or 'document'.")

# import pygame


# if __name__ == "__main__":
#     input_type = input("Enter input type (text/document): ").strip().lower()
    
#     if input_type == "text":
#         text_input = input("Enter text: ")
#         audio_output = process_input(text_input, 'text')  # Generate speech
        
#         # Check if the file exists before playing
#         if os.path.exists(audio_output):
#             pygame.mixer.init()
#             pygame.mixer.music.load(audio_output)
#             pygame.mixer.music.play()
            
#             print("Playing audio...")  # Indicate audio is playing
            
#             while pygame.mixer.music.get_busy():  # Wait for playback to finish
#                 pygame.time.Clock().tick(10)
            
#             print("Playback finished.")
#         else:
#             print(f"Audio file {audio_output} not found.")
        
#     elif input_type == "document":
#         file_path = input("Enter document file path: ").strip()
#         audio_output = process_input(file_path, 'document')
#         # Play the generated audio
        
#         # Check if the file exists before playing
#         if os.path.exists(audio_output):
#             pygame.mixer.init()
#             pygame.mixer.music.load(audio_output)
#             pygame.mixer.music.play()
            
#             print("Playing audio...")  # Indicate audio is playing
            
#             while pygame.mixer.music.get_busy():  # Wait for playback to finish
#                 pygame.time.Clock().tick(10)
            
#             print("Playback finished.")
#         else:
#             print(f"Audio file {audio_output} not found.")
        
#     else:
#         print("Invalid input type. Please enter 'text' or 'document'.")
    
#     print("Speech generation completed.")

import os
import fitz  # PyMuPDF
import re
import io
from PIL import Image
import pytesseract
import uuid
from equation_to_text import MathToSpeech
import pygame
import numpy as np
import torch
import soundfile as sf
import matplotlib.pyplot as plt
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# Initialize components
pygame.mixer.init()
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("Atrishi/speecht5_tts_voxpopuli_nl")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
speaker_embeddings = torch.zeros((1, 512))
math_to_speech = MathToSpeech()

def split_text_into_chunks(text, max_words=30):
    """Split text into chunks of `max_words` length at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        words_in_sentence = sentence.split()
        total_words = len(current_chunk.split()) + len(words_in_sentence)

        if total_words > max_words:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

import uuid

def process_text(text, output_file=None, chunk_size=30):
    """Convert long text to speech and save as WAV file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_segments = []

    # Generate a unique output file if not provided
    if output_file is None:
        output_file = f"output_{uuid.uuid4().hex[:8]}.wav"

    # Split text into chunks at sentence boundaries or word boundaries
    chunks = split_text_into_chunks(text, max_words=chunk_size)

    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i+1}] Synthesizing: {chunk[:60]}...")
        inputs = processor(text=chunk, return_tensors="pt").to(device)
        spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

        with torch.no_grad():
            speech = vocoder(spectrogram)

        audio_segments.append(speech.squeeze().cpu().numpy())

    # Concatenate all chunks into one waveform
    full_audio = np.concatenate(audio_segments, axis=-1)

    # Save final audio
    sf.write(output_file, full_audio, samplerate=16000)
    print(f"\nFull audio saved to {output_file}")
    return output_file




def process_document(file_path):
    """Process document and extract text with equations."""
    try:
        # Handle file paths with spaces
        file_path = file_path.strip('"')  # Remove quotes if present
        if not os.path.exists(file_path):
            print(f"\nError: File not found at path: {file_path}")
            print("Please check the path and try again.")
            return None
            
        extracted_text = []
        doc = fitz.open(file_path)
        
        # Loop through each page in the document
        for page_num, page in enumerate(doc):
            # Get the text blocks (this includes text along with their coordinates)
            text_blocks = page.get_text("dict")["blocks"]
            
            # Get the images from the page (with coordinates)
            images = page.get_images(full=True)

            # Merge text and images by their position on the page
            elements = []

            # Extract and sort text elements
            for block in text_blocks:
                if "lines" in block:  # Check if text lines exist
                    text_content = "\n".join(
                        span["text"] for line in block["lines"] if "spans" in line for span in line["spans"]
                    )
                    elements.append({"type": "text", "content": text_content, "y_pos": block["bbox"][1]})

            image_blocks = [b for b in text_blocks if b.get("type") == 1]
            image_refs = page.get_images(full=True)

            # Ensure equal length (otherwise fallback or warn)
            if len(image_blocks) == len(image_refs):
                for block, img_info in zip(image_blocks, image_refs):
                    xref = img_info[0]
                    y_pos = block["bbox"][1]  # Top Y position of image

                    try:
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(image_bytes))

                        elements.append({"type": "image", "image": image, "y_pos": y_pos})
                    except Exception as e:
                        print(f"Failed to extract image at xref {xref}: {e}")
            else:
                print("Warning: Mismatch between image blocks and xrefs. Skipping image-position linking for this page.")


            # Sort elements by their Y-coordinate (preserve document order)
            elements.sort(key=lambda e: e["y_pos"])

            # Process text and images in order
            for element in elements:
                if element["type"] == "text":
                    text = element["content"]
                    
                    # Detect and replace LaTeX equations
                    latex_equations = re.findall(r"\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]", text)
                    for eq in latex_equations:
                        text = text.replace(eq, f" {math_to_speech.process_latex_equation(eq)}")
                    
                    # Detect and replace text-based mathematical equations
                    text_equations = re.findall(r"[A-Za-z0-9\s\+\-\*/\^=<>,;:!@#\$%&\(\)\{\}\[\]_]+", text)
                    text_equations = [eq.strip() for eq in text_equations if eq.strip()]
                    for eq in text_equations:
                        text = text.replace(eq, f" {math_to_speech.process_text_equation(eq)}")
                
                    extracted_text.append(text)

                elif element["type"] == "image":
                    # Process image (convert equation or caption)
                    extracted_text.append(math_to_speech.process_image(element["image"]))


        # Combine all extracted text and generate speech
        final_text = '\n'.join(extracted_text)
        
        # Convert numbers to text
        final_text = math_to_speech.process_numbers(final_text)
        
        print(final_text)
        if final_text.strip():
            return process_text(final_text)  # Convert extracted text to speech
        else:
            print("No valid text extracted from the document.")
            return None

    except Exception as e:
        print(f"Error processing the document: {e}")
        return None



def play_audio(audio_file):
    try:
        if os.path.exists(audio_file):
            pygame.mixer.init(frequency=16000)
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            print("\nPlaying audio...")

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            print("Playback finished.")
        else:
            print(f"\nError: Audio file not found at {audio_file}")
    except Exception as e:
        print(f"\nError during playback: {str(e)}")


def clean_file_path(path):
    """Clean and verify file path."""
    path = path.strip()
    path = path.strip('"')  # Remove surrounding quotes if present
    if not os.path.exists(path):
        return None
    return path

def main():
    """Main interactive loop."""
    print("\n" + "="*50)
    print("TEXT AND DOCUMENT TO SPEECH CONVERTER")
    print("="*50)
    print("Enter Q at any time to quit\n")
    
    while True:
        print("\nMAIN MENU:")
        print("1. Convert text to speech")
        print("2. Process PDF/document")
        print("Q. Quit program")
        
        choice = input("\nEnter your choice (1/2/Q): ").strip().upper()
        
        if choice == 'Q':
            print("\nThank you for using the converter. Goodbye!")
            break
            
        elif choice == '1':
            while True:
                text = input("\nEnter text to convert (or Q to return to menu):\n> ").strip()
                if text.upper() == 'Q':
                    break
                if text:
                    try:
                        audio_file = process_text(text)
                        play_audio(audio_file)
                    except Exception as e:
                        print(f"\nError processing text: {str(e)}")
                else:
                    print("\nPlease enter some text or Q to return to menu")
        
        elif choice == '2':
            while True:
                file_path = input("\nEnter document path (or Q to return to menu):\n> ").strip()
                if file_path.upper() == 'Q':
                    break
                
                # Clean and verify path
                clean_path = clean_file_path(file_path)
                if not clean_path:
                    print(f"\nFile not found: {file_path}")
                    print("Please check the path and try again.")
                    print("Tip: Enclose paths with spaces in quotes")
                    continue
                
                try:
                    print(f"\nProcessing document: {clean_path}")
                    extracted_text = process_document(clean_path)
                    if extracted_text:
                        print("\nExtracted content preview:")
                        print(extracted_text[:500] + ("..." if len(extracted_text) > 500 else ""))
                        
                        audio_file = process_text(extracted_text)
                        play_audio(audio_file)
                    else:
                        print("\nNo text could be extracted from the document.")
                except Exception as e:
                    print(f"\nError processing document: {str(e)}")
        
        else:
            print("\nInvalid choice. Please enter 1, 2, or Q.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Goodbye!")
    finally:
        pygame.mixer.quit()
        
