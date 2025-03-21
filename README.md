# Speech Generation

## Project Overview
This project aims to generate speech from any provided document. The process involves scanning the document and executing three main functionalities:

1. **Image to Text**
2. **Equations to Text**
3. **Text to Speech**

The extracted text is then passed through a fine-tuned Text-to-Speech (TTS) model to generate speech.

## Implementation Details
We are currently implementing image captioning using the **BLIP Transformer**. The generated captions are further refined using a **GPT Transformer** to improve accuracy and contextual relevance.

### **Workflow**
1. **Document Processing**
   - Extract images from the document.
   - Recognize and extract text, including mathematical equations.

2. **Image Captioning**
   - Use **BLIP Transformer** for initial caption generation.
   - Fine-tune captions with a **GPT Transformer** for improved results.

3. **Speech Generation**
   - Convert refined text to speech using a fine-tuned **TTS model**.

We are fine-tuning the **Speech-to-Text T5 model** from Hugging Face on the **keithito/LJ Speech dataset** available on Hugging Face. The dataset consists of over 10,000 audio-text samples by a single speaker.

## UML Diagrams
Below are UML diagrams representing the workflow of the project:

![UML Diagram 1](https://github.com/Alind1007/Image_captioning/blob/main/uml1.jpg?raw=true)

![UML Diagram 2](https://github.com/Alind1007/Image_captioning/blob/main/uml2.jpg?raw=true)

---
### **Contributors**
- [Alind Sharma, 22113013]
- [Atrishi Jha, 22114014]

For any issues or contributions, feel free to open a pull request or raise an issue!
