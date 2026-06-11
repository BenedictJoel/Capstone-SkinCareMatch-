import os
import shutil
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm # pastikan pip install tqdm

print("1. Menyiapkan Pre-Trained ResNet50 sebagai 'Mata Murni'...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Kita pakai ResNet50 yang murni dari ImageNet, BUKAN model Anda yang salah belajar
base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# Buang layer terakhir (classifier) karena kita cuma mau ambil fitur visualnya (tekstur/warna)
feature_extractor = nn.Sequential(*list(base_model.children())[:-1]).to(device)
feature_extractor.eval()

# Setup gambar (Tidak perlu augmentasi aneh-aneh untuk ekstraksi fitur)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Folder dataset lama Anda yang labelnya masih berantakan
data_dir = 'dataset_acne_grading' 
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

print(f"2. Membaca {len(dataset)} gambar dan mengekstrak fitur visualnya...")
all_features = []
all_paths = []

# Ambil path gambar asli
for path, _ in dataset.samples:
    all_paths.append(path)

with torch.no_grad():
    for images, _ in tqdm(dataloader):
        images = images.to(device)
        features = feature_extractor(images)
        features = features.view(features.size(0), -1) # Ratakan jadi vektor
        all_features.append(features.cpu().numpy())

all_features = np.vstack(all_features)

print("3. Memaksa AI mengelompokkan ulang gambar menjadi 3 Kelompok (K-Means Clustering)...")
# AI akan mencari sendiri 3 pola yang paling membedakan (misal: Bersih, Bruntusan, Meradang)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
new_labels = kmeans.fit_predict(all_features)

print("4. Membuat folder Dataset Baru yang sudah rapi...")
output_dir = 'dataset_baru_auto'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Buat folder cluster
for i in range(3):
    os.makedirs(os.path.join(output_dir, f'Cluster_{i}'))

# Pindahkan gambar ke rumah barunya masing-masing
for path, label in zip(all_paths, new_labels):
    filename = os.path.basename(path)
    # Format nama file ditambahkan nama folder lama agar Anda tahu asalnya darimana
    old_folder = os.path.basename(os.path.dirname(path)) 
    new_filename = f"{old_folder}_{filename}"
    
    dest = os.path.join(output_dir, f'Cluster_{label}', new_filename)
    shutil.copy(path, dest)

print(f"✅ Selesai! Silakan cek folder '{output_dir}'.")
print("TUGAS ANDA SELANJUTNYA: Buka folder Cluster_0, Cluster_1, dan Cluster_2.")
print("Lihat mayoritas isinya, lalu ubah nama foldern nominal tersebut menjadi Level_0, Level_1, atau Level_2 yang sesuai!")