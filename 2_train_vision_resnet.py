import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import numpy as np
import json
from collections import Counter

print("1. Mempersiapkan Transformasi & Membagi Dataset (Train/Val)...")

# Transformasi SUPER KUAT untuk Data Training (Agar AI tahan banting)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30), 
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), 
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Transformasi BERSIH untuk Data Validasi (Hanya resize & normalize, tanpa augmentasi)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'dataset_baru_auto' # GANTI JIKA NAMA FOLDER DATASET ANDA BEDA

# Buat dua objek dataset untuk mengambil transform yang berbeda
full_train_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
full_val_dataset = datasets.ImageFolder(data_dir, transform=val_transforms)

# Pisahkan dataset: 80% Training, 20% Validation
num_data = len(full_train_dataset)
indices = list(range(num_data))
np.random.seed(42) # Agar splitnya selalu sama setiap kali di-run
np.random.shuffle(indices)
split = int(np.floor(0.2 * num_data)) # 20% untuk validasi

train_idx, val_idx = indices[split:], indices[:split]

train_dataset = Subset(full_train_dataset, train_idx)
val_dataset = Subset(full_val_dataset, val_idx)

print(f"Total Data: {num_data} | Training: {len(train_dataset)} | Validation: {len(val_dataset)}")

# ==========================================
# 2. STRATEGI OVERSAMPLING (HANYA UNTUK DATA TRAINING)
# ==========================================
print("2. Menganalisis ketimpangan data Training...")
# Mengambil label hanya dari data training
train_labels = [full_train_dataset.targets[i] for i in train_idx]

class_counts = [0] * len(full_train_dataset.classes)
for label in train_labels:
    class_counts[label] += 1

print(f"Distribusi Training: {dict(zip(full_train_dataset.classes, class_counts))}")

# Membuat bobot sampel
class_weights = [1.0 / count if count > 0 else 0 for count in class_counts]
sample_weights = [class_weights[label] for label in train_labels]

# Dataloader Training (Dipaksa Seimbang dengan Sampler)
train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
train_dataloader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)

# Dataloader Validasi (Normal, Tanpa Sampler, Tanpa Shuffle)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print("Distribusi Batch Training sekarang dipaksa seimbang!")

# ==========================================
# 3. UPGRADE OTAK AI (MEMBUKA KUNCI RESNET50)
# ==========================================
print("3. Membangun model ResNet-50 dan membuka saraf deteksi tekstur")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True 
    else:
        param.requires_grad = False 

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(full_train_dataset.classes))

# ==========================================
# 4. TRAINING & VALIDATION LOOP
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

print(f"4. Mulai proses belajar di {device} (Training 15 Epoch)...")
epochs = 15

for epoch in range(epochs):
    # --- FASE TRAINING ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
    epoch_train_acc = 100 * train_correct / train_total
    epoch_train_loss = train_loss / len(train_dataloader)

    # --- FASE VALIDATION (EVALUASI) ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad(): # Matikan perhitungan gradien agar memori hemat
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    epoch_val_acc = 100 * val_correct / val_total
    epoch_val_loss = val_loss / len(val_dataloader)

    # Cetak Hasil per Epoch
    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

# ==========================================
# 5. SIMPAN MODEL
# ==========================================
torch.save(model.state_dict(), 'skin_vision_resnet.pth')

label_map = {v: k for k, v in full_train_dataset.class_to_idx.items()}
with open('label_map.json', 'w') as f:
    json.dump(label_map, f)

print("✅ Selesai! Model ResNet50 (skin_vision_resnet.pth) super presisi berhasil dibuat!")