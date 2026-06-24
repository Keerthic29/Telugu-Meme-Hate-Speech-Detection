# Telugu Meme Hate Speech Detection

## Overview

AI Content Moderation Studio is an intelligent content moderation system designed to detect abusive, hateful, and offensive content from both text and meme images.

With the increasing use of social media platforms, harmful content is often shared through text as well as images containing embedded text. Traditional moderation systems struggle to analyze such content effectively.

This project combines Optical Character Recognition (OCR), Natural Language Processing (NLP), and Rule-Based Detection techniques to automatically identify harmful content in Telugu and English languages.

---

## Problem Statement

Social media platforms generate a large amount of user content every day. Manual moderation of abusive and hateful content is time-consuming and inefficient.

Additionally, offensive content is frequently shared through memes and images, making traditional text-only moderation systems ineffective.

This project aims to automatically detect abusive and offensive content from both text and meme images using Artificial Intelligence.

---

## Key Features

- Text-based content analysis
- Meme image analysis
- Telugu and English language support
- OCR-based text extraction using EasyOCR
- Rule-based profanity detection
- AI-powered toxicity classification
- Confidence score generation
- Risk level assessment
- Content moderation recommendations
- Interactive web dashboard using Gradio

---

## Technology Stack

### Programming Language
- Python

### Frontend
- Gradio

### OCR Engine
- EasyOCR

### Artificial Intelligence Model
- Toxic-BERT (`unitary/toxic-bert`)

### Deep Learning Framework
- PyTorch

### Machine Learning Library
- Hugging Face Transformers

### Image Processing
- NumPy
- Pillow

### Text Processing
- Regular Expressions (Regex)

---

## System Architecture

```text
User Input (Text / Image)
           │
           ▼
 OCR Extraction (EasyOCR)
           │
           ▼
   Text Normalization
           │
           ▼
Rule-Based Profanity Detection
           │
           ▼
 Toxic-BERT Toxicity Analysis
           │
           ▼
   Hybrid Decision Engine
           │
           ▼
      Risk Assessment
           │
           ▼
     Dashboard Output
```

---

## Project Workflow

### Text Analysis

1. User enters Telugu or English text.
2. Text is normalized and cleaned.
3. Rule-based profanity detection is performed.
4. Toxic-BERT analyzes contextual toxicity.
5. Final risk level and recommendation are generated.

### Meme/Image Analysis

1. User uploads a meme image.
2. EasyOCR extracts text from the image.
3. Extracted text is passed to the moderation engine.
4. Rule-based and AI-based analysis are performed.
5. Final classification results are displayed.

---

## AI Model Used

### Toxic-BERT

**Model Name:** `unitary/toxic-bert`

Toxic-BERT is a transformer-based toxicity classification model available through Hugging Face.

The model predicts multiple toxicity categories including:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Identity Hate
- Insult

The highest toxicity score is used to determine the overall content risk level.

---

## Hybrid Detection Approach

The project combines two detection mechanisms:

### Rule-Based Profanity Engine

Uses predefined Telugu and English profanity dictionaries to detect explicit abusive language.

**Advantages:**
- Fast execution
- Accurate keyword detection
- Handles spelling variations and text normalization

### AI-Based Toxicity Detection

Uses Toxic-BERT to understand contextual meaning and identify offensive language even when explicit abusive words are absent.

**Advantages:**
- Context-aware detection
- Better generalization
- Reduced false negatives

Combining both approaches improves moderation accuracy and reliability.

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd project-folder
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

---

## Dashboard Access

After launching the application, open the local URL provided by Gradio:

```text
http://127.0.0.1:7860
```

---

## Results

- Successfully detects abusive and offensive content in Telugu and English.
- Supports moderation of both text and meme-based content.
- Combines OCR and NLP technologies for improved content analysis.
- Generates confidence scores and moderation recommendations.
- Provides an intuitive dashboard for user interaction.

---

## Future Enhancements

- Support for additional Indian languages.
- Video and audio content moderation.
- Real-time moderation APIs.
- Social media platform integration.
- Advanced contextual meme understanding.
- Cloud deployment for large-scale moderation.

---

## Contributors

- Team Lead – System Architecture, Backend Development, Toxic-BERT Integration, Hybrid Decision Logic
- OCR & Image Processing – EasyOCR Integration, Text Extraction Pipeline
- Frontend & Testing – Gradio Dashboard, Validation, Documentation

---

## License

This project is developed for academic and educational purposes.
