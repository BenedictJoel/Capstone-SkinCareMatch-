import os
import glob
from PIL import Image, ImageEnhance, ImageOps
import random

# ==========================================
# KONFIGURASI FOLDER
# ==========================================
# Arahkan langsung ke folder induk yang membungkus Level_0, Level_1, Level_2
DATA_DIR = 'dataset_acne_grading' 

# ==========================================
# 1. HITUNG JUMLAH DATA SAAT INI
# ==========================================
class_counts = {}
for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)
    if os.path.isdir(folder_path):
        # Cari semua gambar (jpg, jpeg, png) termasuk yang huruf besar
        images = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            images.extend(glob.glob(os.path.join(folder_path, ext)))
            images.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        class_counts[folder] = len(images)

print("📊 Distribusi Data Saat Ini:")
for k, v in class_counts.items():
    print(f" - {k}: {v} gambar")

# ==========================================
# 2. TENTUKAN TARGET JUMLAH DATA
# ==========================================
if not class_counts:
    print("❌ ERROR: Tidak ada folder atau gambar ditemukan! Cek kembali nama folder Anda.")
    exit()

# Cari jumlah data terbanyak (Target)
TARGET_COUNT = max(class_counts.values())
print(f"\n🎯 Target kita: Semua folder kelas harus memiliki {TARGET_COUNT} gambar!")

# ==========================================
# 3. MULAI PROSES AUGMENTASI (PENGGANDAAN)
# ==========================================
augmented_total = 0

for folder, count in class_counts.items():
    if count < TARGET_COUNT:
        folder_path = os.path.join(DATA_DIR, folder)
        
        # Ambil daftar gambar asli di folder ini untuk digandakan
        images = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            images.extend(glob.glob(os.path.join(folder_path, ext)))
            images.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
        needed = TARGET_COUNT - count
        print(f"⏳ Menambah {needed} gambar buatan untuk kelas '{folder}'...")

        for i in range(needed):
            # Pilih gambar acak dari folder tersebut
            img_path = random.choice(images)
            try:
                img = Image.open(img_path).convert('RGB')

                # Beri efek acak (Anti-Overfitting)
                if random.random() > 0.5:
                    img = ImageOps.mirror(img) # Cermin
                
                if random.random() > 0.5:
                    angle = random.uniform(-15, 15)
                    img = img.rotate(angle) # Putar
                
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(random.uniform(0.8, 1.2)) # Gelap/Terang

                # Simpan gambar baru ke folder yang sama
                base_name = os.path.basename(img_path)
                new_img_name = f"aug_{i}_{base_name}"
                new_img_path = os.path.join(folder_path, new_img_name)
                img.save(new_img_path)
                augmented_total += 1
                
            except Exception as e:
                continue

print(f"\n✅ PROSES PENYEIMBANGAN DATA SELESAI! (Berhasil membuat {augmented_total} gambar baru)")
print("Sekarang semua kelas memiliki jumlah gambar yang seimbang dan siap dilatih.")