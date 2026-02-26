# 🧪 Custom Model Training Guide — Image Captioning from Scratch

This guide explains how to train your **own** image captioning model instead of using the pretrained BLIP model. This approach gives you full control over the architecture, training data, and model behavior.

---

## 📊 Pretrained vs Custom Model Comparison

| Aspect | Pretrained (BLIP) | Custom (Train from Scratch) |
|---|---|---|
| **Setup Time** | 5 minutes (pip install) | Days–weeks (training) |
| **Caption Quality** | Excellent (trained on 14M+ images) | Depends on dataset & training |
| **Dataset Required** | None | Yes (Flickr8k/30k/COCO) |
| **GPU Required** | No (CPU works) | Yes (strongly recommended) |
| **Customizability** | Limited to prompts | Full control |
| **Model Size** | ~1.5GB (large) / ~500MB (base) | You decide |
| **Best For** | Production apps, demos | Learning, research, niche domains |

---

## 🏗️ Architecture: CNN Encoder + Transformer Decoder

### Overview
```
Input Image (224×224)
    │
    ▼
┌──────────────────┐
│  CNN Encoder      │   ← ResNet-50 or EfficientNet (pretrained on ImageNet)
│  (Feature         │   ← Extract 2048-dim feature vectors
│   Extraction)     │   ← Remove final classification layer
└────────┬─────────┘
         │ Image Features (2048-dim)
         │
    Linear Projection
         │ Mapped to (512-dim)
         │
         ▼
┌──────────────────┐
│  Transformer      │   ← 6 layers, 8 attention heads
│  Decoder          │   ← Cross-attention to image features
│  (Text Generation)│   ← Autoregressive (word-by-word)
└────────┬─────────┘
         │
         ▼
    Vocabulary Projection (softmax over vocab_size)
         │
         ▼
    Generated Caption: "a dog playing in the park"
```

---

## 📦 Dataset Preparation

### Recommended Datasets

| Dataset | Images | Captions/Image | Total Captions | Download |
|---|---|---|---|---|
| **Flickr8k** | 8,000 | 5 | 40,000 | [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) |
| **Flickr30k** | 31,000 | 5 | 155,000 | [Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) |
| **MS COCO** | 330,000 | 5 | 1.5M | [cocodataset.org](https://cocodataset.org) |

### Data Preprocessing Steps

```python
import os
import json
from PIL import Image
from collections import Counter
import torchvision.transforms as transforms

# ── Step 1: Load Captions ──
# Flickr8k provides a text file with format: image_id#caption_number \t caption_text
def load_flickr8k_captions(captions_file):
    """
    Parse Flickr8k captions file into a dictionary.
    
    Each image has 5 different captions written by different annotators.
    This diversity helps the model learn varied descriptions.
    """
    captions = {}
    with open(captions_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                image_id = parts[0].split('#')[0]  # Remove caption number
                caption = parts[1].lower().strip()
                if image_id not in captions:
                    captions[image_id] = []
                captions[image_id].append(caption)
    return captions


# ── Step 2: Build Vocabulary ──
# We need to convert words to numerical IDs for the model.
def build_vocabulary(captions_dict, min_word_freq=3):
    """
    Build a word-to-index vocabulary from all captions.
    
    - <PAD>: Padding token (index 0) — fills shorter sequences
    - <SOS>: Start-of-sentence (index 1) — signals the decoder to begin
    - <EOS>: End-of-sentence (index 2) — signals when to stop generating
    - <UNK>: Unknown word (index 3) — replaces rare/unseen words
    
    Words appearing fewer than min_word_freq times are replaced with <UNK>
    to reduce vocabulary size and improve generalization.
    """
    word_counts = Counter()
    for caps in captions_dict.values():
        for cap in caps:
            word_counts.update(cap.split())
    
    # Keep only words that appear at least min_word_freq times
    vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    vocab += [word for word, count in word_counts.items() if count >= min_word_freq]
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return word2idx, idx2word, len(vocab)


# ── Step 3: Image Transforms ──
# CNN encoders expect normalized 224×224 images with ImageNet statistics.
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),         # Random crop for data augmentation
    transforms.RandomHorizontalFlip(),  # Mirror images randomly
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],     # ImageNet mean
        std=[0.229, 0.224, 0.225]       # ImageNet std
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),      # No augmentation for validation
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

---

## 🧠 Model Architecture (PyTorch)

```python
import torch
import torch.nn as nn
import torchvision.models as models
import math


class ImageEncoder(nn.Module):
    """
    CNN-based image encoder using a pretrained ResNet-50.
    
    Why ResNet-50?
    - Pretrained on ImageNet (1.2M images, 1000 classes)
    - Already understands visual features (edges, textures, objects)
    - We remove the final classification layer and use the 2048-dim features
    - These features capture WHAT is in the image and WHERE
    
    The 2048-dim features are projected to the decoder's hidden dimension (512)
    via a linear layer, making them compatible with the Transformer decoder.
    """
    
    def __init__(self, embed_dim=512):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        
        # Remove the final fully-connected classification layer
        # We keep everything up to the average pooling layer
        modules = list(resnet.children())[:-1]  # Remove fc layer
        self.resnet = nn.Sequential(*modules)
        
        # Project from ResNet's 2048-dim to the decoder's embed_dim
        self.projection = nn.Linear(2048, embed_dim)
        self.dropout = nn.Dropout(0.3)
        
        # Freeze early ResNet layers (they capture universal features)
        # Fine-tune only the last few layers for our specific task
        for param in list(self.resnet.parameters())[:-20]:
            param.requires_grad = False
    
    def forward(self, images):
        """
        Args:
            images: Batch of images [batch_size, 3, 224, 224]
        Returns:
            features: Image embeddings [batch_size, 1, embed_dim]
        """
        features = self.resnet(images)        # [batch, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [batch, 2048]
        features = self.dropout(features)
        features = self.projection(features)  # [batch, embed_dim]
        features = features.unsqueeze(1)      # [batch, 1, embed_dim]
        return features


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for the Transformer decoder.
    
    Since Transformers process all tokens simultaneously (no inherent order),
    we must explicitly inject position information. We use sinusoidal functions:
    
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    This encoding has nice properties:
    - Each position gets a unique encoding
    - The model can learn to attend to relative positions
    - It generalizes to sequence lengths not seen during training
    """
    
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CaptionDecoder(nn.Module):
    """
    Transformer decoder for autoregressive text generation.
    
    This decoder generates captions word-by-word using:
    1. SELF-ATTENTION: Each word attends to all previous words (masked)
    2. CROSS-ATTENTION: Each word attends to the image features
    3. FEED-FORWARD: Non-linear transformation at each position
    
    The decoder is "autoregressive" because each word depends on all
    previous words. During training, we use "teacher forcing" — feeding
    the ground-truth previous words. During inference, we feed the model's
    own predictions.
    """
    
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # Standard 4× expansion
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Project hidden states to vocabulary probabilities
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, captions, image_features, tgt_mask=None):
        """
        Args:
            captions: Token IDs [batch_size, seq_len]
            image_features: From encoder [batch_size, 1, embed_dim]
            tgt_mask: Causal mask to prevent attending to future words
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Embed caption tokens + add positional encoding
        caption_embeds = self.word_embedding(captions) * math.sqrt(self.embed_dim)
        caption_embeds = self.pos_encoding(caption_embeds)
        
        # Generate causal mask (lower triangular) if not provided
        if tgt_mask is None:
            seq_len = captions.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
            tgt_mask = tgt_mask.to(captions.device)
        
        # Transformer decoder with cross-attention to image features
        # memory = image_features (what the decoder "looks at" from the image)
        # tgt = caption_embeds (the text sequence being generated)
        output = self.transformer_decoder(
            tgt=caption_embeds,
            memory=image_features,
            tgt_mask=tgt_mask,
        )
        
        logits = self.fc_out(self.dropout(output))
        return logits


class ImageCaptioningModel(nn.Module):
    """
    Complete image captioning model: CNN Encoder + Transformer Decoder.
    """
    
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim)
        self.decoder = CaptionDecoder(vocab_size, embed_dim, num_heads, num_layers)
    
    def forward(self, images, captions, tgt_mask=None):
        image_features = self.encoder(images)
        logits = self.decoder(captions, image_features, tgt_mask)
        return logits
```

---

## 📈 Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model, train_loader, val_loader, vocab_size, num_epochs=30, lr=3e-4):
    """
    Training loop for the image captioning model.
    
    Key concepts:
    - Cross-Entropy Loss: Measures how well predicted word probabilities
      match the actual next word. Lower = better.
    - Teacher Forcing: During training, we feed the GROUND TRUTH previous
      word (not the model's prediction) to speed up learning.
    - Learning Rate Scheduling: Gradually reduce LR for fine-tuning.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <PAD> tokens
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            
            # Input:  <SOS> word1 word2 ... wordN
            # Target: word1 word2 ... wordN <EOS>
            input_captions = captions[:, :-1]   # Remove last token
            target_captions = captions[:, 1:]   # Remove first token (<SOS>)
            
            # Forward pass
            logits = model(images, input_captions)
            
            # Reshape for cross-entropy: [batch*seq_len, vocab_size] vs [batch*seq_len]
            loss = criterion(
                logits.reshape(-1, vocab_size),
                target_captions.reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        val_loss = validate(model, val_loader, criterion, device, vocab_size)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_caption_model.pth')
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")


def validate(model, val_loader, criterion, device, vocab_size):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions in val_loader:
            images = images.to(device)
            captions = captions.to(device)
            input_captions = captions[:, :-1]
            target_captions = captions[:, 1:]
            logits = model(images, input_captions)
            loss = criterion(logits.reshape(-1, vocab_size), target_captions.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)
```

---

## 📏 Evaluation Metrics

```python
from collections import Counter
import math

def compute_bleu(reference, candidate, max_n=4):
    """
    Compute BLEU (Bilingual Evaluation Understudy) score.
    
    BLEU measures how many n-grams in the generated caption match the
    reference caption. It's the standard metric for text generation:
    
    - BLEU-1: Unigram overlap (individual word matches)
    - BLEU-2: Bigram overlap (two-word phrase matches)
    - BLEU-3: Trigram overlap
    - BLEU-4: 4-gram overlap (most commonly reported)
    
    Scores range from 0 to 1 (higher is better).
    Typical good model: BLEU-4 ≈ 0.25-0.35 on COCO.
    """
    scores = []
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    for n in range(1, max_n + 1):
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        cand_ngrams = Counter([tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens)-n+1)])
        
        matches = sum((cand_ngrams & ref_ngrams).values())
        total = max(sum(cand_ngrams.values()), 1)
        
        scores.append(matches / total)
    
    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1)))
    
    # Geometric mean of n-gram precisions
    log_avg = sum(math.log(max(s, 1e-10)) for s in scores) / max_n
    bleu = bp * math.exp(log_avg)
    
    return bleu
```

---

## 🔄 Integrating Your Custom Model into the App

To use your trained model instead of BLIP, modify `models/caption_engine.py`:

```python
# Replace the BLIP loading code with:
class CaptionEngine:
    def __init__(self):
        self.model = ImageCaptioningModel(vocab_size=YOUR_VOCAB_SIZE)
        self.model.load_state_dict(torch.load('best_caption_model.pth'))
        self.model.eval()
        # Load your vocabulary
        self.word2idx = load_vocab('vocab.json')
        self.idx2word = {v: k for k, v in self.word2idx.items()}
    
    def generate_caption(self, image, max_length=50):
        # Preprocess image
        img_tensor = val_transform(image).unsqueeze(0)
        features = self.model.encoder(img_tensor)
        
        # Autoregressive generation
        tokens = [self.word2idx['<SOS>']]
        for _ in range(max_length):
            input_ids = torch.tensor([tokens])
            logits = self.model.decoder(input_ids, features)
            next_token = logits[0, -1, :].argmax().item()
            if next_token == self.word2idx['<EOS>']:
                break
            tokens.append(next_token)
        
        caption = ' '.join([self.idx2word[t] for t in tokens[1:]])
        return caption
```

---

## 📋 Summary

| What You Learn | Where |
|---|---|
| CNN feature extraction | `ImageEncoder` class |
| Transformer decoder | `CaptionDecoder` class |
| Cross-attention | `TransformerDecoderLayer` (built-in) |
| Positional encoding | `PositionalEncoding` class |
| Teacher forcing | Training loop (input vs target split) |
| Beam search | BLIP's `generate()` method |
| BLEU evaluation | `compute_bleu()` function |

> **Recommendation**: Start with the pretrained BLIP approach for a working app, then use this guide to understand and build your own model for learning purposes.
