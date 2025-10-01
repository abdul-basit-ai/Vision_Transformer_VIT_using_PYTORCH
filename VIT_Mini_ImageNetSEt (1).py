

import os
import shutil
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

#  device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters ---
#  ViT on Tiny ImageNet
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 3e-4
NUM_CLASSES = 200
IMAGE_SIZE = 64
PATCH_SIZE = 16
EMBED_DIM = 256
NUM_HEADS = 8
DEPTH = 6
MLP_DIM = 512
DROPOUT = 0.1
CHANNELS = 3

#  Dataset and DataLoader
print("\n--- Downloading and Preparing Tiny ImageNet ---")
!wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
!unzip -q tiny-imagenet-200.zip
data_dir = './tiny-imagenet-200'

# Organize the validation data
val_dir = os.path.join(data_dir, 'val')
val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
val_img_dict = {}
with open(val_annotations_file, 'r') as f:
    for line in f.readlines():
        words = line.strip().split('\t')
        val_img_dict[words[0]] = words[1]
for img_filename, label in val_img_dict.items():
    src = os.path.join(val_dir, 'images', img_filename)
    dst = os.path.join(val_dir, label, img_filename)
    if not os.path.exists(os.path.join(val_dir, label)):
        os.makedirs(os.path.join(val_dir, label))
    if os.path.exists(src):
        shutil.move(src, dst)
shutil.rmtree(os.path.join(val_dir, 'images'))
os.remove(val_annotations_file)
print("Validation set successfully organized.")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
}
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# ViT Model Arch
class PatchEmbeddings(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.projection(x).flatten(2).transpose(1, 2)
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.dropout(x)
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, embed_dim, dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, embed_dim, depth, num_heads, mlp_dim, dropout):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = PatchEmbeddings(in_channels, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        cls_token_output = x[:, 0]
        logits = self.head(cls_token_output)
        return logits

# Training and Evaluation Functions
def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss, total_correct = 0, 0
    with tqdm(dataloader, unit="batch") as tepoch:
        for images, labels in tepoch:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            tepoch.set_postfix(loss=total_loss / (tepoch.n+1), accuracy=100. * total_correct / len(dataloader.dataset))
    return total_loss / len(dataloader), 100. * total_correct / len(dataloader.dataset)
def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    with torch.no_grad(), tqdm(dataloader, unit="batch") as tepoch:
        for images, labels in tepoch:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            tepoch.set_postfix(accuracy=100. * total_correct / len(dataloader.dataset))
    return 100. * total_correct / len(dataloader.dataset)

# Loop
print("\n--- Initializing Model and Training ---")
model = VisionTransformer(
    img_size=IMAGE_SIZE, patch_size=PATCH_SIZE, in_channels=CHANNELS, num_classes=NUM_CLASSES,
    embed_dim=EMBED_DIM, depth=DEPTH, num_heads=NUM_HEADS, mlp_dim=MLP_DIM, dropout=DROPOUT
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
    test_acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

print("\nTraining complete!")