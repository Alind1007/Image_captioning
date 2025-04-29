

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
from transformers import VitsModel, VitsTokenizer
import time
# Initialize components
pygame.mixer.init()
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model1 = SpeechT5ForTextToSpeech.from_pretrained("Atrishi/speecht5_tts_voxpopuli_nl")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
speaker_embeddings=torch.zeros((1,512))

# Load VITS model for better quality and more stable voice
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

math_to_speech = MathToSpeech()

#from TTS.api import TTS

# # Load XTTS model (multilingual + multi-speaker)
# tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

# # Path to your reference audio sample
# reference_audio_path = r"C:\Users\atris\Downloads\Siri_example.mp3"

# # Extract speaker embedding from sample
# speaker_latents = tts.get_conditioning_latents(audio_path="reference.wav")
# speaker_embedding = speaker_latents["c_latent"]  # This is the actual 512-dim vector
# math_to_speech = MathToSpeech()
def ensure_letters_are_read(text):
    """
    Capitalize isolated lowercase letters (excluding 'a') and add a period after them
    to help TTS models pronounce them clearly.
    """
    import re

    # Pattern to match isolated lowercase letters except 'a'
    pattern = r'(?<!\w)([b-z])(?!\w)'

    # Replace with uppercase letter and period
    processed_text = re.sub(pattern, lambda m: m.group(1).upper() + '.', text)

    return processed_text



def split_text_into_chunks(text, max_words=25):
    """Improved text splitting that maintains better context and avoids mid-sentence breaks."""
    # First split by paragraphs
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    
    for para in paragraphs:
        # Split paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', para)
        current_chunk = []
        word_count = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_word_count = len(words)
            
            # If adding this sentence would exceed max_words, finalize current chunk
            if word_count + sentence_word_count > max_words and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                word_count = 0
            
            current_chunk.append(sentence)
            word_count += sentence_word_count
        
        # Add remaining sentences in this paragraph
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
    return chunks


import uuid

def process_text1(text, output_file=None, chunk_size=30):
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
        spectrogram = model1.generate_speech(inputs["input_ids"], speaker_embeddings)

        with torch.no_grad():
            speech = vocoder(spectrogram)

        audio_segments.append(speech.squeeze().cpu().numpy())

    # Concatenate all chunks into one waveform
    full_audio = np.concatenate(audio_segments, axis=-1)

    # Save final audio
    sf.write(output_file, full_audio, samplerate=16000)
    print(f"\nFull audio saved to {output_file}")
    return output_file


def process_text(text, output_file=None, chunk_size=25):
    """Convert text to speech with improved voice quality and chunk handling."""
    # Generate a unique output file if not provided
    if output_file is None:
        output_file = f"output_{int(time.time())}.wav"
    
    text = ensure_letters_are_read(text)
    
    # Split text into chunks
    chunks = split_text_into_chunks(text, max_words=chunk_size)
    audio_segments = []
    
    print(f"Processing {len(chunks)} text chunks...")
    
    for i, chunk in enumerate(chunks):
        try:
            print(f"Processing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            
            # Tokenize and generate speech
            inputs = tokenizer(chunk, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(**inputs)
            
            # Convert to numpy array and normalize
            speech = output.waveform.squeeze().cpu().numpy()
            speech = speech / np.max(np.abs(speech))  # Normalize
            
            # Add small silence between chunks (0.1s)
            silence = np.zeros(int(0.1 * 16000))  # 16000 is sample rate
            audio_segments.append(speech)
            audio_segments.append(silence)
            
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            continue
        
    if not audio_segments:
        print("No audio was generated.")
        return None
    
    # Concatenate all chunks with proper spacing
    full_audio = np.concatenate(audio_segments)
    
    # Apply simple fade in/out to avoid clicks
    fade_samples = 500
    full_audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
    full_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # Save final audio
    sf.write(output_file, full_audio, samplerate=model.config.sampling_rate)
    print(f"Audio saved to {output_file}")
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
                    print("Image found")
                    extracted_text.append(math_to_speech.process_image(element["image"]))


        # Combine all extracted text and generate speech
        final_text = '\n'.join(extracted_text)
        
        print(final_text)
        # Convert numbers to text
        final_text = math_to_speech.process_numbers(final_text)
        
        print(final_text)
        if final_text.strip():
            return process_text1(final_text)  # Convert extracted text to speech
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
        
