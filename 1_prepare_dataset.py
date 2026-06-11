import pandas as pd
import numpy as np
import re
import pickle
import os
import glob
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords

# 1. Download stopwords jika belum ada
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))
custom_stopwords = {'aqua', 'water', 'extract', 'ekstrak', 'dan', 'yang', 'untuk', 'dengan'}
stop_words.update(custom_stopwords)

print("1. Membaca Semua File CSV di folder SkinSAFE...")
# Gunakan glob untuk mencari semua file berakhiran .csv di dalam folder SkinSAFE
folder_path = 'SkinSAFE'
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

df_list = []
for file in all_files:
    try:
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
    except Exception as e:
        print(f"Gagal membaca {file}: {e}")

# Gabungkan seluruh potongan dataset menjadi 1 DataFrame utuh
df_mentah = pd.concat(df_list, ignore_index=True)
print(f"Total seluruh produk yang berhasil digabung: {len(df_mentah)} baris")

# 2. Standarisasi Nama Kolom
df = pd.DataFrame()
df['product_name'] = df_mentah['product_name']
df['brand'] = df_mentah['brand'] 
df['product_type'] = df_mentah['category'] 
df['ingredients'] = df_mentah['ingredients'].fillna('')
df['image_url'] = df_mentah['image_url'].fillna('')

# Berdasarkan cuplikan file 1-200.csv, sepertinya tidak ada kolom "description".
# Jadi kita akali dengan menggunakan nama produk sebagai deskripsi dasarnya.
if 'description' in df_mentah.columns:
    df['description'] = df_mentah['description'].fillna('')
else:
    df['description'] = df_mentah['product_name'].fillna('')

# Buat dummy harga secara random (Rp 50.000 - Rp 350.000)
np.random.seed(42)
df['price'] = np.random.randint(50, 350, df.shape[0]) * 1000

# 3. Filter Hanya Kategori Skincare (Membuang kategori yang tidak relevan)
skincare_keywords = 'cleanser|toner|serum|moisturizer|sunscreen|cream|lotion|mask|wash|exfoliat'
df = df[df['product_type'].astype(str).str.lower().str.contains(skincare_keywords, na=False)]

# Kita ambil sampel 3.000 produk agar pemrosesan tidak memakan waktu berjam-jam,
# namun tetap cukup kaya untuk sistem rekomendasi.
sample_size = min(3000, len(df))
df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
print(f"Total produk setelah difilter & disampel: {len(df)}")

# 4. Fungsi Pembersihan Teks
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', '', str(text))
    words = text.lower().split()
    cleaned_words = [w for w in words if w not in stop_words]
    return ' '.join(cleaned_words)

print("2. Membersihkan Teks Ingredients & Description...")
df['desc_clean'] = df['description'].apply(clean_text)
df['ingred_clean'] = df['ingredients'].apply(clean_text)

# Gabungkan fitur untuk dibaca oleh model NLP
df['combined_features'] = df['desc_clean'] + " " + df['ingred_clean']

# 5. Load Model & Generate Embeddings
print("3. Memuat Model NLP dan Membuat Vektor Semantik (Ini memakan waktu beberapa menit)...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
semantic_matrix = model.encode(df['combined_features'].tolist(), show_progress_bar=True)

# 6. Simpan File Database Baru
print("4. Menyimpan file df_final_v2.pkl dan semantic_matrix_v2.pkl...")
df.to_pickle('df_final_v2.pkl')
with open('semantic_matrix_v2.pkl', 'wb') as f:
    pickle.dump(semantic_matrix, f)

print("✅ STEP 1 SELESAI! Matriks SkinSAFE siap digunakan.")