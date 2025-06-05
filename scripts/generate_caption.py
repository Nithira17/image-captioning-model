import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from transformers import AutoTokenizer
from PIL import Image
from models import VisionEncoderDecoder, TokenDrop
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from models.utils import SampleCaption

import matplotlib
matplotlib.use('TkAgg')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model(checkpoint_path, device):
    image_size = 128
    hidden_size = 192
    num_layers = (6, 6)
    num_heads = 8
    patch_size = 8

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    caption_model = VisionEncoderDecoder(
        image_size=image_size,
        channels_in=3,
        num_emb=tokenizer.vocab_size,
        patch_size=patch_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    caption_model.load_state_dict(checkpoint['model_state_dict'])
    caption_model.eval()

    return caption_model, tokenizer

# Generate caption for a single image
def generate_caption(image_path, model, tokenizer, device, temp=0.5):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    sos_token = torch.tensor([[101]], device=device)
    log_tokens = [sos_token]

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            image_embedding = model.encoder(image_tensor)

        for i in range(50):
            input_tokens = torch.cat(log_tokens, dim=1)
            
            # Decode the input tokens into the next predicted tokens
            data_pred = model.decoder(input_tokens.to(device), image_embedding)
            
            # Sample from the distribution of predicted probabilities
            dist = torch.distributions.Categorical(logits=data_pred[:, -1]/temp)
            next_tokens = dist.sample().reshape(1, 1)

            # Append the nextpredicted token to the sequence
            log_tokens.append(next_tokens.cpu())

            # End of caption is predicted
            if next_tokens.item() == 102:
                break

    # Decode the tokens to text
    pred_text = torch.cat(log_tokens, dim=1)
    pred_text_strings = tokenizer.decode(pred_text[0], skip_special_tokens=True)
    pred_text = "".join(pred_text_strings)

    # Display the image and caption
    plt.figure(figsize=(8, 8))
    plt.imshow(make_grid(image_tensor.cpu(), normalize=True).permute(1, 2, 0))
    plt.axis("off")
    plt.title(pred_text)
    plt.show()

    return pred_text

def test_on_dataset(index, checkpoint_path):
    data_set_root = '/home/nithira/ImageCaptioning/Datasets'
    validation_set = 'val2017'
    validation_set_path = os.path.join(data_set_root, validation_set)
    validation_ann_file = '{}/annotations/captions_{}.json'.format(data_set_root, validation_set)

    val_transform = transforms.Compose([transforms.Resize(128),
                                transforms.CenterCrop(128),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
    
    val_dataset = datasets.CocoCaptions(root=validation_set_path,
                                     annFile=validation_ann_file,
                                     transform=val_transform,
                                     target_transform=SampleCaption())
    
    data_loader_val = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)

    dataiter = next(iter(data_loader_val))
    test_images, test_captions = dataiter

    index = index
    test_image = test_images[index].unsqueeze(0)

    plt.figure(figsize=(3, 3))
    out = torchvision.utils.make_grid(test_image, 1, normalize=True)

    plt.imshow(out.numpy().transpose((1, 2, 0)))
    print(test_captions[index])

    sos_token = 101*torch.ones(1, 1).long()
    temp = 0.5

    log_tokens = [sos_token]
    model, tokenizer = load_model(checkpoint_path, device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            image_embedding = model.encoder(test_image.to(device))

        for i in range(50):
            input_tokens = torch.cat(log_tokens, 1)

            data_pred = model.decoder(input_tokens.to(device), image_embedding)

            dist = torch.distributions.Categorical(logits=data_pred[:, -1]/temp)
            next_tokens = dist.sample().reshape(1, 1)

            log_tokens.append(next_tokens.cpu())

            if next_tokens.item() == 102:
                break

    pred_text = torch.cat(log_tokens, 1)
    pred_text_strings = tokenizer.decode(pred_text[0], skip_special_tokens=True)
    pred_text = "".join(pred_text_strings)

    plt.figure(figsize=(3, 3))
    out = torchvision.utils.make_grid(test_image, 1, normalize=True)
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    print(pred_text)

if __name__ == "__main__":
    # Set paths
    checkpoint_path = '/home/nithira/ImageCaptioning/checkpoints/captioning_model.pt'
    test_image_path = ""

    test_on_dataset(28, checkpoint_path)


