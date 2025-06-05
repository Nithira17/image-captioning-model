import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from models import VisionEncoderDecoder, TokenDrop, SampleCaption
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
lr = 1e-4
image_size = 128
nepochs = 120
batch_size = 128

# Dataset paths
data_set_root = '/home/nithira/ImageCaptioning/Datasets'
train_set = 'train2017'
validation_set = 'val2017'

train_set_path = os.path.join(data_set_root, train_set)
train_ann_file = '{}/annotations/captions_{}.json'.format(data_set_root, train_set)

validation_set_path = os.path.join(data_set_root, validation_set)
validation_ann_file = '{}/annotations/captions_{}.json'.format(data_set_root, validation_set)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Data Transforms
train_transform = transforms.Compose([transforms.Resize(image_size),
                                      transforms.RandomCrop(image_size),
                                      transforms.AutoAugment(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])]) 

val_transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])]) 

# Datasets
train_dataset = datasets.CocoCaptions(root=train_set_path,
                                      annFile=train_ann_file,
                                      transform=train_transform,
                                      target_transform=SampleCaption())

val_dataset = datasets.CocoCaptions(root=validation_set_path,
                                     annFile=validation_ann_file,
                                     transform=val_transform,
                                     target_transform=SampleCaption())

# Data Loaders
data_loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
data_loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Model
hidden_size = 192
num_layers = (6, 6)
num_heads = 8
patch_size = 8

caption_model = VisionEncoderDecoder(image_size=image_size, 
                                     channels_in=3, 
                                     num_emb=tokenizer.vocab_size, 
                                     patch_size=patch_size, 
                                     num_layers=num_layers, 
                                     hidden_size=hidden_size, 
                                     num_heads=num_heads
                                    ).to(device)

# Optimizer, Loss function, Gradient Scaler
optimizer = optim.Adam(caption_model.parameters(), lr=lr)
scaler = torch.cuda.amp.GradScaler()
loss_fn = nn.CrossEntropyLoss(reduction="none")

# Token Drop
td = TokenDrop(prob=0.5)

eval_loss_logger = []

def evaluate(model, data_loader, loss_fn, device):
    model.eval()

    with torch.no_grad():
        for images, captions in tqdm(data_loader_val, desc="Eval", leave=False):
            images = images.to(device)

            # Tokenize captions
            tokens = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
            token_ids = tokens['input_ids'].to(device)
            padding_mask = tokens['attention_mask'].to(device)
            
            # Prepare target tokens
            bs = token_ids.shape[0]
            target_ids = torch.cat((token_ids[:, 1:],
                                    torch.zeros(bs, 1, device=device).long()), 1)
            
            # Forward pass
            with torch.cuda.amp.autocast():
                pred = model(images, token_ids, padding_mask=padding_mask)

            # Compute the loss
            loss_mask = (~(target_ids == 0)).float()
            loss = (loss_fn(pred.transpose(1, 2), target_ids) * loss_mask).sum()/loss_mask.sum()

            return loss.item()

def train():
    # Load checkpoint if available
    checkpoint_path = '/home/nithira/ImageCaptioning/checkpoints/captioning_model.pt'
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        caption_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        training_loss_logger = checkpoint.get('training_data_logger', [])
        #eval_loss_logger = checkpoint.get('eval_data_logger', [])
        start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
        print(f"Resuming training from epoch {start_epoch}...")
    else:
        print("No checkpoint found. Starting training from scratch.")
        training_loss_logger = []
        eval_loss_logger = []
    
    caption_model.train()

    for epoch in trange(start_epoch, nepochs, leave=False, desc="Epoch"):
        for images, captions in tqdm(data_loader_train, desc="Training", leave=False):
            images = images.to(device)

            # Tokenize captions
            tokens = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
            token_ids = tokens['input_ids'].to(device)
            padding_mask = tokens['attention_mask'].to(device)

            # Prepare target tokens
            bs = token_ids.shape[0]
            target_ids = torch.cat((token_ids[:, 1:],
                                    torch.zeros(bs, 1, device=device).long()), 1)
            
            # Apply token drop
            tokens_in = td(token_ids)

            with torch.cuda.amp.autocast():
                # Forward pass
                pred = caption_model(images, tokens_in, padding_mask=padding_mask)

            # Compute the loss
            loss_mask = (~(target_ids == 0)).float()
            loss = (loss_fn(pred.transpose(1, 2), target_ids) * loss_mask).sum()/loss_mask.sum()


            # Backpropagation
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Log the training loss
            training_loss_logger.append(loss.item())

        print(f"Epoch {epoch + 1}/{nepochs}, Loss: {loss.item()}")

        # Evaluation
        eval_loss_logger.append(evaluate(caption_model, data_loader_val, loss_fn, device))
        print(f"Validation Loss: {eval_loss_logger[-1]}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': caption_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_data_logger': training_loss_logger,
            'eval_data_logger': eval_loss_logger
        }

        torch.save(checkpoint, f'/home/nithira/ImageCaptioning/checkpoints/captioning_model_epoch_{epoch + 1}.pt')

if __name__ == "__main__":
    train()

