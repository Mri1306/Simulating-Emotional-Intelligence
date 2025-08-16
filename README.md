# Simulating-Emotional-Intelligence
This repository contains the code, dataset, and methodology for the paper:  
📄 Paper DOI/Link: *(https://www.researchgate.net/profile/Veronica-Mangiaterra-2/publication/394273313_On_choosing_the_vehicles_of_metaphors_without_a_body_evidence_from_Large_Language_Models/links/6895fde8c345306d43cc2922/On-choosing-the-vehicles-of-metaphors-without-a-body-evidence-from-Large-Language-Models.pdf#page=87)*  
---

## 📌 Overview
Human emotions emerge from a rich interplay of verbal, para-verbal, and non-verbal cues. Large Language Models (LLMs) often struggle to replicate this nuance.  
This project introduces a **dual-path framework** that enhances emotional intelligence in LLMs by combining:

- **Behavioral Conditioning**: Fine-tuning on metadata-rich prompts (tone, response time, body language).  
- **Analogical Retrieval**: A Retrieval-Augmented Generation (RAG) system filtering responses by emotional metadata.  

The core dataset, **MECC (Multimodal Emotionally Conditioned Corpus)**, provides a unique resource for training emotionally intelligent systems.

---

## 📂 Dataset: MECC

**MECC** contains **1,764 question-answer pairs** from 31 participants, annotated across **15 emotional categories** with behavioral metadata.  

### Emotion Categories
- **Primary Emotions**: Love & Affection, Anger & Frustration, Fear & Anxiety, Happiness & Joy, Sadness & Grief, Guilt & Regret, Loneliness & Isolation  
- **Self-Reflective Cognition**: Confidence & Self-Belief, Decision-Making, Forgiveness & Letting Go, Emotional Growth & Self-Reflection  
- **Social-Affective Constructs**: Empathy & Understanding Others, Gratitude & Contentment, Stress & Coping, Non-Verbal Communication  

### Metadata Captured
- **Tone** (calm, hesitant, defensive, etc.)  
- **Response Time** (fast, moderate, slow)  
- **Body Language** (gestures, gaze, posture)  

---

## 🏗 Methodology

### 1. Behavioral Prompt Construction
Structured interview data flattened into JSON prompts with embedded behavioral cues.  

### 2. Fine-Tuning
- **Base Model**: LLaMA-3.1–8B–Instruct  
- **Technique**: Parameter-Efficient Fine-Tuning (LoRA) with 8-bit NF4 quantization.  
- **Config**:  
  - LoRA rank = 8  
  - Alpha = 16  
  - Epochs = 3  
  - Batch size = 8 (accumulated)  
  - Learning rate = 5e-5  

### 3. Emotionally Aligned RAG
- **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformers)  
- **Indexing**: FAISS with metadata (emotion, tone, response time, body language).  
- **Retrieval**: Top-k exemplar retrieval with joint scoring (semantic + emotional + behavioral).  

### 4. Dual-Path Inference
- **Path 1 (Fine-Tuned Generation)**: Directly conditions LLaMA model on behavioral prompts.  
- **Path 2 (RAG-Enhanced Generation)**: Augments user query with retrieved exemplars for context-sensitive responses.  

## 📊 Results

### Performance Comparison
| Metric | RAG Model | Non-RAG Model |
|--------|-----------|---------------|
| Emotional Accuracy | 38.2% | 39.9% |
| BERTScore F1 | 0.827 | 1.000 |
| Cosine Similarity | 0.443 | 1.000 |
| Pearson Correlation | 0.152 (p=0.004) | 0.372 (p=0.000) |
| MSE | **0.091** | 15.78 |
| Perplexity | 5.55 | 4.38 |

- **RAG** produced **more diverse and human-like emotional responses**, even though raw accuracy was slightly lower.  
- **Non-RAG** showed higher textual similarity but risked overfitting and reduced variability.  


