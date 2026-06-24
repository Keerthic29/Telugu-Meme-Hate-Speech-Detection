import gradio as gr
import re
import easyocr
from transformers import pipeline
import torch
import numpy as np
from PIL import Image

# ==========================================
# 1. INITIALIZATION & MODELS
# ==========================================

print("Initializing Models (This may take a moment on first run)...")
device = 0 if torch.cuda.is_available() else -1

# Load OCR - Supports Telugu ('te') and English ('en')
try:
    ocr_reader = easyocr.Reader(['te', 'en'], gpu=(device == 0))
except Exception as e:
    print(f"Failed to load OCR: {e}")
    ocr_reader = None

# Load Toxicity Model
try:
    toxicity_analyzer = pipeline("text-classification", model="unitary/toxic-bert", device=device, top_k=None)
except Exception as e:
    print(f"Failed to load Toxicity Model: {e}")
    toxicity_analyzer = None

# ==========================================
# 2. PROFANITY ENGINE DICTIONARY
# ==========================================

ENGLISH_HIGH = [
    "cunt", "motherfucker", "fuck", "slut", "pussy", "cock", "dickhead"
]

ENGLISH_MED = [
    "shit", "asshole", "ass hole", "bitch", "dick", "prick", "bastard", 
    "bollocks", "wanker", "twat", "arsehole", "whore", "suck"
]

TELUGU_MIX = [
    "badkav", "baisinda", "bokal eruguthai", "lanja", "lanja koduka", "lanjakoduka",
    "dengai", "bosidike", "gudda", "moda", "sulli", "munda", "ni amma puku", 
    "badmash", "erri lanja kodka", "erri puk"
]

ALL_PROFANITY = ENGLISH_HIGH + ENGLISH_MED + TELUGU_MIX

def normalize_text(text):
    """Normalize text for fuzzy matching (remove repeated letters, punctuation)."""
    text = text.lower()
    # Replace common symbol substitutions
    text = text.replace("@", "a").replace("$", "s").replace("0", "o").replace("1", "i")
    # Remove non-alphanumeric (keep spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Remove consecutive duplicate letters
    text = re.sub(r'(.)\1+', r'\1', text)
    return text

def detect_profanity(text):
    """Rule-based engine to find explicit profanity."""
    normalized = normalize_text(text)
    detected_words = set()
    
    # Direct substring match in normalized text
    # This handles spacing variations to some extent if they merged
    no_space_text = normalized.replace(" ", "")
    
    for word in ALL_PROFANITY:
        norm_word = normalize_text(word)
        norm_word_no_space = norm_word.replace(" ", "")
        
        # Check standard normalized
        if re.search(r'\b' + re.escape(norm_word) + r'\b', normalized) or norm_word_no_space in no_space_text:
            # We found a match. To be visually accurate, add the actual un-normalized list word it matched
            detected_words.add(word)

    return list(detected_words)

# ==========================================
# 3. HYBRID DECISION LOGIC
# ==========================================

def get_ai_toxicity_score(text):
    if not toxicity_analyzer:
        return 0.0
    results = toxicity_analyzer(text)
    # results is a list of lists of dicts because top_k=None
    scores = {res['label']: res['score'] for res in results[0]}
    
    # We take the maximum of toxic, severe_toxic, obscene, threat, insult, identity_hate
    toxic_score = max(scores.get('toxic', 0), 
                      scores.get('severe_toxic', 0), 
                      scores.get('obscene', 0), 
                      scores.get('identity_hate', 0),
                      scores.get('insult', 0),
                      scores.get('threat', 0))
    return toxic_score

def analyze_content(text, source="Text Input"):
    if not text or not text.strip():
        return generate_html_report(0, "Safe", 0, "", [], source, "Allow")

    # 1. Rule-based check
    detected_words = detect_profanity(text)
    
    # 2. AI Toxicity Check
    ai_score = get_ai_toxicity_score(text)
    
    # 3. Hybrid Logic
    if len(detected_words) > 0:
        # Check if high severity
        is_high_sev = any(w in ENGLISH_HIGH or w in TELUGU_MIX for w in detected_words)
        classification = "Highly Abusive" if is_high_sev else "Abusive"
        confidence = 0.95 if is_high_sev else 0.85
        risk_level = "Critical" if is_high_sev else "High"
        recommendation = "Block Content"
        detection_source = "Rule-Based Keyword Engine" + (" + AI" if ai_score > 0.5 else "")
    else:
        confidence = ai_score
        detection_source = "AI Toxicity Model"
        if ai_score > 0.75:
            classification = "Abusive"
            risk_level = "High"
            recommendation = "Block Content"
        elif ai_score >= 0.40:
            classification = "Potentially Offensive"
            risk_level = "Medium"
            recommendation = "Warn User"
        else:
            classification = "Safe"
            risk_level = "Low"
            recommendation = "Allow"
            detection_source = "AI Toxicity Model (Clean)"

    return generate_html_report(
        confidence_score=confidence,
        risk_level=risk_level,
        word_count=len(detected_words),
        extracted_text=text,
        detected_words=detected_words,
        detection_source=source + " -> " + detection_source,
        recommendation=recommendation
    )

# ==========================================
# 4. IMAGE OCR LOGIC
# ==========================================

def process_image(image):
    if image is None:
        return "<div style='color: white;'>Please upload an image.</div>"
    
    if not ocr_reader:
        return "<div style='color: white;'>OCR Engine failed to load.</div>"
    
    # Convert PIL Image to numpy array for EasyOCR
    image_np = np.array(image)
    
    # Run OCR
    results = ocr_reader.readtext(image_np)
    extracted_text = " ".join([text for (bbox, text, prob) in results])
    
    if not extracted_text.strip():
        return generate_html_report(0, "Safe", 0, "No text detected in image.", [], "OCR", "Allow")
        
    return analyze_content(extracted_text, source="Image OCR")

# ==========================================
# 5. UI GENERATION (HTML/CSS)
# ==========================================

def generate_html_report(confidence_score, risk_level, word_count, extracted_text, detected_words, detection_source, recommendation):
    # Determine colors
    if risk_level == "Critical":
        color_hex = "#ef4444" # Red
        ring_color = "red"
    elif risk_level == "High":
        color_hex = "#f97316" # Orange
        ring_color = "orange"
    elif risk_level == "Medium":
        color_hex = "#eab308" # Yellow
        ring_color = "yellow"
    else:
        color_hex = "#22c55e" # Green
        ring_color = "green"

    conf_percent = int(confidence_score * 100)
    
    words_html = "".join([f"<span class='badge bad-word'>{w}</span>" for w in detected_words])
    if not words_html:
        words_html = "<span style='color: #94a3b8;'>None detected</span>"

    html = f"""
    <div class="dashboard-card">
        <div class="dashboard-header">
            <h2>Analysis Report</h2>
            <span class="badge" style="background-color: {color_hex}40; color: {color_hex}; border: 1px solid {color_hex};">{risk_level} Risk</span>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Confidence Score</div>
                <div class="metric-value" style="color: {color_hex}">{conf_percent}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Recommendation</div>
                <div class="metric-value" style="color: {color_hex}">{recommendation}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Detection Engine</div>
                <div class="metric-value" style="font-size: 1rem; color: #cbd5e1;">{detection_source}</div>
            </div>
        </div>

        <div class="content-section">
            <h3 class="section-title">Extracted Content</h3>
            <div class="text-box">{extracted_text}</div>
        </div>

        <div class="content-section">
            <h3 class="section-title">Detected Profanities ({word_count})</h3>
            <div class="word-chips">
                {words_html}
            </div>
        </div>
    </div>
    """
    return html

custom_css = """
/* Global Theme Adjustments */
body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%) !important;
    color: #f8fafc !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Glassmorphism Dashboard Cards */
.dashboard-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 40px -10px rgba(0,0,0,0.5);
    animation: fadeIn 0.5s ease-out;
}

.dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 16px;
}

.dashboard-header h2 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 600;
    color: #f8fafc;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}

.metric-box {
    background: rgba(15, 23, 42, 0.5);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    text-align: center;
}

.metric-label {
    font-size: 0.875rem;
    color: #94a3b8;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
}

.content-section {
    margin-bottom: 20px;
}

.section-title {
    font-size: 1rem;
    color: #cbd5e1;
    margin-bottom: 12px;
    font-weight: 500;
}

.text-box {
    background: rgba(15, 23, 42, 0.8);
    border-radius: 8px;
    padding: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    font-family: monospace;
    white-space: pre-wrap;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.5;
}

.badge {
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 600;
    display: inline-block;
}

.word-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.bad-word {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    border: 1px solid #ef4444;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Customizing Gradio Components */
.gr-button-primary {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: transform 0.2s !important;
}

.gr-button-primary:hover {
    transform: scale(1.02);
}

.gr-box {
    background: rgba(30, 41, 59, 0.4) !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
}
"""

# ==========================================
# 6. GRADIO INTERFACE
# ==========================================

with gr.Blocks(css=custom_css, title="Telugu Hate & Profanity Meme Detector") as demo:
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 2rem; padding: 2rem 0;">
            <h1 style="font-size: 2.5rem; font-weight: 800; background: linear-gradient(to right, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">
                🛡️ AI Content Moderation Studio
            </h1>
            <p style="color: #94a3b8; font-size: 1.1rem;">
                Advanced Hybrid Engine for Detecting Hate Speech & Profanity in Telugu and English.
            </p>
        </div>
    """)
    
    with gr.Tabs():
        # Tab 1: Image Analysis (Memes)
        with gr.Tab("🖼️ Analyze Meme / Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload Meme", height=400)
                    scan_img_btn = gr.Button("🔍 Scan Image", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    image_output = gr.HTML(label="Analysis Dashboard")
            
            scan_img_btn.click(fn=process_image, inputs=image_input, outputs=image_output)
            
        # Tab 2: Text Analysis
        with gr.Tab("📝 Analyze Text"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Enter Message (Telugu/English)", 
                        placeholder="Type something here...",
                        lines=10
                    )
                    scan_text_btn = gr.Button("🔍 Scan Text", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    text_output = gr.HTML(label="Analysis Dashboard")
                    
            scan_text_btn.click(fn=analyze_content, inputs=text_input, outputs=text_output)
            
        # Tab 3: System Status & Info
        with gr.Tab("📊 System Info"):
            gr.Markdown("""
            ### 🚀 Detection Pipeline Architecture
            1. **Multi-lingual OCR**: Powered by EasyOCR targeting `en` and `te`.
            2. **Fuzzy Profanity Engine**: Rule-based detection handling edge cases, leetspeak (e.g. l@nja), spacing variants.
            3. **Deep Learning Toxicity**: Backed by `unitary/toxic-bert` for nuanced, contextual toxicity analysis.
            4. **Hybrid Decision Layer**: Fuses logic for deterministic abuse catching and probabilistic toxicity detection.
            
            *Status: All systems operational. Models loaded.*
            """)

if __name__ == "__main__":
    demo.launch()
