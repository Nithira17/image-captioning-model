import torch
from models import VisionEncoderDecoder
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 128
hidden_size = 192
num_layers = (6, 6)
num_heads = 8
patch_size = 8

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# Build model
model = VisionEncoderDecoder(
    image_size=image_size,
    channels_in=3,
    num_emb=tokenizer.vocab_size,
    patch_size=patch_size,
    num_layers=num_layers,
    hidden_size=hidden_size,
    num_heads=num_heads
).to(device)
checkpoint = torch.load("captioning_model.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Dummy inputs
dummy_image = torch.randn(1, 3, 128, 128)
dummy_seq = torch.ones(1, 1, dtype=torch.long) * 101
dummy_mask = torch.ones(1, 1, dtype=torch.long)

torch.onnx.export(
    model,
    (dummy_image, dummy_seq, dummy_mask),
    "captioning_model.onnx",
    input_names=["image", "seq", "mask"],
    output_names=["output"],
    opset_version=13
)

print("ONNX model exported as captioning_model.onnx")
