import os
import glob
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

print("Membaca seluruh file dataset CSV di folder SkinSAFE...")

# 1. BACA SEMUA FILE CSV DAN GABUNGKAN
folder_path = 'SkinSAFE'
# Mencari semua file berakhiran .csv di dalam folder SkinSAFE
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

if not all_files:
    print(f"❌ ERROR: Tidak ada file CSV ditemukan di folder '{folder_path}'. Pastikan foldernya ada!")
    exit()

df_list = []
for filename in all_files:
    try:
        temp_df = pd.read_csv(filename)
        df_list.append(temp_df)
    except Exception as e:
        print(f"Gagal membaca {filename}: {e}")

# Gabungkan semuanya jadi satu tabel raksasa
df = pd.concat(df_list, axis=0, ignore_index=True)
print(f"✅ Berhasil menggabungkan {len(all_files)} file CSV!")
print(f"Total produk keseluruhan: {len(df)} produk.")

# 2. BERSIHKAN DATA (CLEANING)
df = df.dropna(subset=['product_name', 'ingredients']).copy()
df['brand'] = df['brand'].fillna('Unknown')
df['category'] = df['category'].fillna('General')
df['usage_type'] = df['usage_type'].fillna('General')

# (Opsional) Beri harga default jika kolom harga tidak ada agar web tidak error
if 'price' not in df.columns:
    import random
    df['price'] = [random.randint(50000, 300000) for _ in range(len(df))]

# 3. GABUNGKAN FITUR (CATEGORY & USAGE TYPE MASUK KE SINI)
df['combined_features'] = (
    "Usage: " + df['usage_type'] + ". " +
    "Category: " + df['category'] + ". " +
    "Brand: " + df['brand'] + ". " +
    "Ingredients: " + df['ingredients']
)

# 4. PROSES NLP (TRANSFORMER)
print("Loading NLP Model (MiniLM)...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Mengubah teks ke vektor angka (Proses ini mungkin butuh waktu beberapa menit karena datanya banyak)...")
semantic_matrix = model.encode(df['combined_features'].tolist())

# 5. SIMPAN DATABASE VERSI 3
df.to_pickle('df_final_v3.pkl')
with open('semantic_matrix_v3.pkl', 'wb') as f:
    pickle.dump(semantic_matrix, f)

print("✅ Selesai! Otak NLP versi terbaru (df_final_v3.pkl dan semantic_matrix_v3.pkl) berhasil dibuat dengan ribuan data!")