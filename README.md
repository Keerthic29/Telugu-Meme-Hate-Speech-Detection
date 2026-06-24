# Telugu Hate & Profanity Meme Detector

A production-quality AI web application designed to act as a modern SaaS cybersecurity dashboard for detecting hateful, abusive, and profane content in both text and uploaded images (memes) using Telugu and English.

## Features
* **Hybrid Detection Engine**: Combines a rigorous Rule-Based Profanity Engine with an AI Toxicity Model (`unitary/toxic-bert`).
* **OCR Support**: Extracts text from images in Telugu and English using `EasyOCR`.
* **Dynamic SaaS Dashboard**: Premium dark-themed UI featuring glassmorphism, animated gradients, and real-time scanning analytics.
* **Smart Filtering**: Detects complex profanities, including repeated letters, partial matches, and spaced words.

## Architecture
1. **User Input**: Image Upload or Text Entry.
2. **OCR Extraction**: Uses EasyOCR for uploaded memes.
3. **Rule-Based Engine**: Matches normalized text against high-severity and medium-severity English/Telugu profanity lists.
4. **AI Toxicity Model**: Uses `toxic-bert` for nuanced hate speech and toxicity scoring.
5. **Hybrid Logic**: Rules take precedence to ensure critical slurs are never missed. AI catches subtler forms of toxicity.
6. **Result Dashboard**: Visually displays Risk Level, Confidence, Extracted Text, and Detection Source.

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```
*Note: The first time you run the application, it will download the EasyOCR models and the Toxic-BERT model from Hugging Face.*

### 3. Access the Dashboard
Open your browser and navigate to the local URL provided by Gradio (usually `http://127.0.0.1:7860`).

## Hugging Face Spaces Deployment
To deploy this project to Hugging Face Spaces:
1. Create a new Space and select **Gradio** as the SDK.
2. Upload `app.py`, `requirements.txt`, and this `README.md`.
3. The Space will automatically install the dependencies and launch the application.
