import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import json
import os

# ==========================================
# KONFIGURASI FOLDER
# ==========================================
# Arahkan ke folder yang SAMA dengan skrip balancer tadi
DATA_DIR = 'dataset_acne_grading'  

device = torch.device("cpu")
print(f"Menggunakan device: {device}")

# ==========================================
# 1. PERSIAPAN DATA & 80/20 SPLIT
# ==========================================
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Muat seluruh dataset (PyTorch otomatis membaca Level_0, Level_1, Level_2 sebagai kelas)
try:
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=image_transforms)
except Exception as e:
    print(f"❌ ERROR: Tidak dapat memuat dataset. Pastikan folder {DATA_DIR} ada. Detail: {e}")
    exit()

# Buat dan simpan label_map.json secara otomatis
label_map = {v: k for k, v in full_dataset.class_to_idx.items()}
with open('label_map.json', 'w') as f:
    json.dump(label_map, f)
print(f"✅ Berhasil memetakan kelas ke label_map.json: {label_map}")

# Potong dataset: 80% untuk Train, 20% untuk Validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Masukkan ke DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"\nTotal Data Keseluruhan : {len(full_dataset)} gambar")
print(f"Data untuk Belajar (Train) : {len(train_dataset)} gambar (80%)")
print(f"Data untuk Ujian (Val)     : {len(val_dataset)} gambar (20%)\n")

# ==========================================
# 2. BANGUN MODEL MOBILENETV2
# ==========================================
print("Membangun arsitektur MobileNetV2...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Bekukan layer dasar agar tidak merusak otak yang sudah pintar dari ImageNet
for param in model.parameters():
    param.requires_grad = False

# Ubah output akhir menyesuaikan jumlah level jerawat (3 kelas)
num_classes = len(full_dataset.classes)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# ==========================================
# 3. PROSES TRAINING
# ==========================================
epochs = 5 # Set 5 epoch dulu agar tidak terlalu lama

print("Memulai proses training...")
for epoch in range(epochs):
    # Mode Latihan
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    train_acc = 100 * correct / total
    
    # Mode Ujian (Validasi)
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# Simpan otak AI-nya!
torch.save(model.state_dict(), 'skin_vision_model.pth')
print("\n✅ MODEL BERHASIL DISIMPAN! (File: skin_vision_model.pth)")