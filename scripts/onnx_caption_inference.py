import argparse
from PIL import Image
import numpy as np
import onnxruntime
import json
import time

from transformers import AutoTokenizer

# Image pre-processing
def crop_resize(image, new_size):
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    upper = (height - min_dim) // 2
    right = left + min_dim
    lower = upper + min_dim
    square_image = image.crop((left, upper, right, lower))
    resized_image = square_image.resize((new_size, new_size))
    return resized_image

def image_normalise_reshape(image, mean, std):
    h, w, c = image.shape
    image = image.transpose((2, 0, 1)) / 255.0
    np_means = np.array(mean).reshape(c, 1, 1)
    np_stds = np.array(std).reshape(c, 1, 1)
    norm_image = (image - np_means) / (np_stds + 1e-6)
    return np.expand_dims(norm_image, 0).astype(np.float32)

def generate_caption_onnx(session, image_arr, tokenizer, max_len=50, temp=1.0):
    sos_token = 101
    eos_token = 102
    seq = np.array([[sos_token]], dtype=np.int64)
    mask = np.ones_like(seq, dtype=np.int64)
    caption = []

    for _ in range(max_len):
        ort_inputs = {
            session.get_inputs()[0].name: image_arr,
            session.get_inputs()[1].name: seq,
            session.get_inputs()[2].name: mask,
        }
        ort_outs = session.run(None, ort_inputs)
        logits = ort_outs[0]  
        next_token_logits = logits[0, -1] / temp
        next_token = int(np.argmax(next_token_logits))
        if next_token == eos_token:
            break
        caption.append(next_token)
        seq = np.concatenate([seq, [[next_token]]], axis=1)
        mask = np.concatenate([mask, [[1]]], axis=1)
    full_tokens = [sos_token] + caption
    text = tokenizer.decode(full_tokens, skip_special_tokens=True)
    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Caption an image using an ONNX model on Raspberry Pi")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model file")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (jpg/png)")
    args = parser.parse_args()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_size = 128  

    # Load tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Preprocess image
    img = Image.open(args.image).convert("RGB")
    img = crop_resize(img, image_size)
    img_np = np.array(img)
    img_input = image_normalise_reshape(img_np, mean, std)

    # Load ONNX model
    session = onnxruntime.InferenceSession(args.model, providers=['CPUExecutionProvider'])

    # Inference and captioning
    start = time.time()
    caption = generate_caption_onnx(session, img_input, tokenizer)
    end = time.time()

    print(f"Caption: {caption}")
    print("Inference time: %.3fs" % (end - start))

if __name__ == "__main__":
    main()
