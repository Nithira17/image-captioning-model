# 🖼️ Image Captioning from Scratch using Transformers

This project showcases an **image captioning model built entirely from scratch using PyTorch**, where **every key component of the Transformer decoder was manually implemented** — no reliance on pre-trained transformer models or libraries like Hugging Face Transformers for modeling.

It’s a complete end-to-end pipeline using the **MS COCO dataset**, integrating **computer vision and natural language processing** to generate human-like captions for images. The goal was to **deepen understanding of attention mechanisms, sequence modeling, and multimodal learning** by implementing the full architecture manually.

---

## 🚀 What This Project Does

- Takes an input image and generates a natural-language caption.
- Uses a CNN (e.g., ResNet) to encode the image into dense feature vectors.
- Uses a **custom-built Transformer-based decoder** to generate captions word-by-word.
- Trains and evaluates the full model on the **MS COCO** dataset using **cross-entropy loss** and **batch evaluation**.
- Uses **Hugging Face tokenizers** for text preprocessing and vocab management.

---

## 🧠 Key Components

1. ✅ **Custom Data Loader**  
   Loads and processes image–caption pairs from the MS COCO dataset, sampling one caption per image and applying transformations.

2. ✅ **CNN-Based Encoder**  
   Extracts fixed-size feature representations from images using a pretrained CNN backbone (e.g., ResNet-50), removing the final classification layer.

3. ✅ **Transformer Decoder (Manual Implementation)**  
   Built entirely from scratch based on the “Attention Is All You Need” paper. Includes:
   - Scaled Dot-Product Attention
   - Multi-Head Attention
   - Positional Encoding
   - Masked Self-Attention
   - Add & Norm layers
   - Feedforward layers

4. ✅ **Caption Tokenization with Hugging Face**  
   Utilized `transformers` tokenizers for caption preprocessing, padding, and numericalization, ensuring compatibility with vocabulary generation.

5. ✅ **Training Loop & Evaluation**  
   - Implemented custom training loop with teacher forcing and masking
   - Cross-entropy loss with padding token ignored
   - Evaluated using batch-based inference and BLEU scores

---

## 🔍 Why This Project Matters

Unlike most image captioning projects that use pretrained transformers or encoder-decoder libraries, **this project implements the full Transformer decoder logic from the ground up** to gain a deeper understanding of:

- How attention and sequence modeling work internally
- The integration of image features with language generation
- Handling multimodal data in a training loop
- Challenges in training transformers from scratch

---


