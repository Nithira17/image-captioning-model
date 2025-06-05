# Real-Time Image Captioning on Raspberry Pi Using ONNX

## Project Overview

This project demonstrates an end-to-end pipeline for **automatic image captioning** using deep learning, optimized for deployment on affordable hardware—the Raspberry Pi 4B. Leveraging PyTorch for model development and ONNX for efficient cross-platform inference, this system takes a static input image and produces a human-readable caption. The project highlights the synergy of computer vision and natural language processing in embedded systems.

---


## Workflow

### 1. Model Development (on PC)
- **Model**: Transformer-based encoder-decoder for image captioning
- **Training**: Uses PyTorch, COCO dataset, and HuggingFace tokenizer
- **Scripts**: `models/`, `train.py`, `generate_caption.py`

### 2. Export to ONNX (on PC)
- **Why**: ONNX makes the model hardware-agnostic, smaller, and faster for inference
- **How**: Script `export_to_onnx.py` exports the trained PyTorch model to `captioning_model.onnx`

### 3. Raspberry Pi Deployment
- **Copy**: Transfer `captioning_model.onnx` and `onnx_caption_inference.py` to the Pi
- **Install**: Required Python libraries (`onnxruntime`, `transformers`, `Pillow`)
- **Usage**: Provide an image file and run the inference script; receive a caption in the terminal



