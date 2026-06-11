from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import cv2
import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json

app = Flask(__name__)

# ==========================================
# 1. LOAD MODEL VISION (Acne Severity)
# ==========================================
print("Loading Vision Model (Acne Severity)...")
device = torch.device("cpu")

with open('label_map.json', 'r') as f:
    label_map_reverse = {int(k): v for k, v in json.load(f).items()}

vision_model = models.mobilenet_v2(weights=None)
vision_model.classifier[1] = nn.Linear(vision_model.last_channel, len(label_map_reverse))
vision_model.load_state_dict(torch.load('skin_vision_model.pth', map_location=device))
vision_model.eval() 

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ==========================================
# 2. LOAD MODEL NLP & DATASET SKINSAFE
# ==========================================
print("Loading NLP Model & Dataset...")
df_final = pd.read_pickle('df_final_v3.pkl')
with open('semantic_matrix_v3.pkl', 'rb') as f:
    semantic_matrix = pickle.load(f)
nlp_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ==========================================
# 3. FUNGSI TRANSLATOR NLP
# ==========================================
def extract_search_rules(user_text):
    user_text = user_text.lower()
    rules = {"product_types": [], "concerns": []}
    
    if any(k in user_text for k in ["sabun", "cuci muka", "pembersih"]): rules["product_types"].extend(["cleans", "wash", "soap"])
    if any(k in user_text for k in ["pelembap", "krim", "lotion"]): rules["product_types"].extend(["moisturiz", "cream", "lotion"])
    if any(k in user_text for k in ["toner", "penyegar"]): rules["product_types"].extend(["toner", "astringent"])
    if any(k in user_text for k in ["serum", "ampoule", "essence"]): rules["product_types"].extend(["serum", "ampoule", "essence"])
    if any(k in user_text for k in ["sunscreen", "uv", "tabir surya", "matahari", "cahaya"]): rules["product_types"].extend(["sunscreen", "uv", "spf"])

    if any(k in user_text for k in ["jerawat", "bruntusan", "acne"]): rules["concerns"].extend(["acne", "salicylic", "benzoyl", "tea tree", "blemish"])
    if any(k in user_text for k in ["keriput", "kerut", "tua", "penuaan", "aging"]): rules["concerns"].extend(["anti-aging", "wrinkle", "retinol", "peptide", "collagen"])
    if any(k in user_text for k in ["kusam", "flek", "hitam", "bekas", "cerah"]): rules["concerns"].extend(["brightening", "vitamin c", "niacinamide", "arbutin", "dark spot"])
    if any(k in user_text for k in ["kering", "mengelupas"]): rules["concerns"].extend(["hydrat", "moist", "hyaluronic", "ceramide", "dry skin"])
    if any(k in user_text for k in ["minyak", "berminyak", "pori", "komedo"]): rules["concerns"].extend(["oil", "sebum", "pore", "blackhead"])
    if any(k in user_text for k in ["sensitif", "merah", "iritasi"]): rules["concerns"].extend(["sensitive", "redness", "sooth", "calm", "centella"])
    
    return rules

def generate_dynamic_reason(user_text, combined_features, ingredients):
    stop_words = {'saya', 'aku', 'butuh', 'ingin', 'mau', 'cari', 'untuk', 'yang', 'dan', 'di', 'pada', 'buat', 'muka', 'wajah', 'kulit', 'tolong'}
    user_words = [w.lower() for w in str(user_text).split() if len(w) > 3 and w.lower() not in stop_words]
    desc_full = (str(combined_features) + " " + str(ingredients)).lower()
    
    matched_words = []
    for uw in user_words:
        if uw in desc_full:
            matched_words.append(uw)
            
    if matched_words:
        return f"Cocok dengan keluhan: {', '.join(set(matched_words))}."
    return "Skor kecocokan semantik (NLP) tinggi."

# ==========================================
# 4. ROUTES FLASK
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_face', methods=['POST'])
def detect_face():
    try:
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # PERBAIKAN OPENCV: Mencegah deteksi mata dengan minSize=(100,100) dan minNeighbors=6
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=(150, 150) # HANYA DETEKSI KOTAK YANG UKURANNYA BESAR (WAJAH UTUH)
        )
        
        if len(faces) == 0:
            return jsonify({"success": False, "error": "Wajah utuh tidak ditemukan. Pastikan kamera lurus menghadap wajah."})
            
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # Tambahkan sedikit margin agar seluruh pipi masuk (Penting untuk deteksi jerawat)
        margin = int(w * 0.1)
        y1, y2 = max(0, y - margin), min(img_cv.shape[0], y + h + margin)
        x1, x2 = max(0, x - margin), min(img_cv.shape[1], x + w + margin)
        
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 3) 
        
        face_crop = img_cv[y1:y2, x1:x2]
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        
        img_tensor = torch.unsqueeze(image_transforms(face_pil), 0).to(device) # type: ignore
        with torch.no_grad():
            outputs = vision_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, predicted = torch.max(probabilities, 0)
            detected_raw_class = label_map_reverse[int(predicted.item())]
        
        mapping_keparahan = {
            "Level_0": "Level 0 (Wajah Bersih / Normal)",
            "Level_1": "Level 1 (Jerawat Ringan / Bruntusan)",
            "Level_2": "Level 2 (Jerawat Kompleks / Meradang)"
        }
        hasil_terjemahan = mapping_keparahan.get(detected_raw_class, detected_raw_class)
        
        text_label = f"{detected_raw_class} ({top_prob.item()*100:.1f}%)"
        cv2.putText(img_cv, text_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', img_cv)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{img_base64}"
            
        return jsonify({
            "success": True, 
            "detection": hasil_terjemahan,
            "raw_class": detected_raw_class,
            "image_with_boxes": img_data_url
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    input_text = data.get('keluhan', '') 
    min_price = int(data.get('min_price', 0))
    max_price = int(data.get('max_price', 1000000))
    allergies = data.get('allergies', [])
    
    df_test = df_final.copy()
    
    parts = input_text.split('| AI mendeteksi')
    user_raw_text = parts[0].strip()
    camera_text = "| AI mendeteksi" + parts[1] if len(parts) > 1 else ""

    rules = extract_search_rules(user_raw_text)
    
    if "level 1" in camera_text.lower():
        rules["concerns"].extend(["acne", "salicylic acid"])
    elif "level 2" in camera_text.lower():
        rules["concerns"].extend(["acne", "benzoyl peroxide", "centella"])

    if not rules["product_types"] and not rules["concerns"] and not camera_text:
        return jsonify({"error": "Maaf, sistem tidak dapat menemukan konteks wajah/skincare dalam keluhan Anda. Tolong tuliskan jenis produk atau masalah kulit Anda."})

    df_test = df_test[(df_test['price'] >= min_price) & (df_test['price'] <= max_price)]
    if allergies:
        for alergen in allergies:
            df_test = df_test[~df_test['ingredients'].str.lower().str.contains(alergen.lower(), na=False)]
            
    if df_test.empty:
        return jsonify({"error": "Tidak ada produk yang sesuai dengan budget/filter Anda."})

    if rules["product_types"]:
        pattern_pt = "|".join(rules["product_types"])
        mask_pt = (
            df_test['product_name'].str.contains(pattern_pt, case=False, na=False) |
            df_test['category'].str.contains(pattern_pt, case=False, na=False) |
            df_test['usage_type'].str.contains(pattern_pt, case=False, na=False)
        )
        if mask_pt.sum() > 0:
            df_test = df_test[mask_pt]

    if rules["concerns"]:
        pattern_c = "|".join(rules["concerns"])
        mask_c = (
            df_test['ingredients'].str.contains(pattern_c, case=False, na=False) |
            df_test['category'].str.contains(pattern_c, case=False, na=False) |
            df_test['product_name'].str.contains(pattern_c, case=False, na=False)
        )
        if mask_c.sum() > 0:
            df_test = df_test[mask_c]

    if df_test.empty:
        return jsonify({"error": "Produk spesifik yang Anda cari tidak tersedia dalam database kami."})

    search_query = user_raw_text.lower() + " " + " ".join(rules["product_types"]) + " ".join(rules["concerns"])
    
    row_positions = [df_final.index.get_loc(idx) for idx in df_test.index]
    user_vector = np.array(nlp_model.encode([search_query])) 
    df_test['sim_score'] = cosine_similarity(user_vector, semantic_matrix[row_positions]).flatten()
    
    df_test['final_score'] = df_test['sim_score'] * 100
    df_sorted = df_test.sort_values(by='final_score', ascending=False)
    
    list_pagi, list_malam = [], []
    used_names = set()
    
    def to_dict(row, score, step_name):
        harga_format = f"Rp {int(row.get('price', 0)):,}".replace(',', '.')
        alasan = generate_dynamic_reason(user_raw_text, row.get('combined_features', ''), row.get('ingredients', ''))
        tipe_produk = f"{row.get('category', row.get('usage_type', '-'))} | 💡 {alasan}"
        
        return {
            "step": step_name,
            "name": row.get('product_name', 'Unknown'),
            "brand": row.get('brand', '-'),
            "type": tipe_produk,
            "price": harga_format,
            "image": row.get('image_url', ''),
            "score": round(min(score, 99.9), 1)
        }

    for _, row in df_sorted.iterrows():
        p_name = row.get('product_name', '')
        p_cat = str(row.get('category', '')) + " " + str(row.get('usage_type', ''))
        p_cat = p_cat.lower()
        desc_ingred = str(row.get('ingredients', '')).lower()
        
        step = "Perawatan Tambahan"
        if 'cleans' in p_cat or 'wash' in p_cat or 'soap' in p_cat: step = "Langkah 1 (Sabun Muka)"
        elif 'toner' in p_cat or 'astringent' in p_cat: step = "Langkah 2 (Toner)"
        elif 'serum' in p_cat or 'ampoule' in p_cat or 'essence' in p_cat: step = "Langkah 3 (Serum)"
        elif 'moisturiz' in p_cat or 'cream' in p_cat or 'lotion' in p_cat: step = "Langkah 4 (Pelembap)"
        elif 'sunscreen' in p_cat or 'uv' in p_cat or 'spf' in p_name.lower(): step = "Langkah 5 (Sunscreen)"

        item = to_dict(row, row.get('final_score', 0), step)
        
        is_sunscreen = 'sunscreen' in p_cat or 'uv' in p_cat or 'spf' in p_name.lower()
        is_night_active = any(k in desc_ingred for k in ['retinol', 'aha', 'peeling', 'glycolic', 'lactic'])

        if is_sunscreen and len(list_pagi) < 5:
            list_pagi.append(item)
            used_names.add(p_name)
        elif is_night_active and len(list_malam) < 5:
            list_malam.append(item)
            used_names.add(p_name)
        else:
            if len(list_pagi) < 5 and p_name not in used_names:
                list_pagi.append(item)
                used_names.add(p_name)
            elif len(list_malam) < 5 and p_name not in used_names:
                list_malam.append(item)
                used_names.add(p_name)
                
        if len(list_pagi) == 5 and len(list_malam) == 5:
            break

    list_pagi = sorted(list_pagi, key=lambda x: x['step'])
    list_malam = sorted(list_malam, key=lambda x: x['step'])
                
    return jsonify({"success": True, "pagi": list_pagi, "malam": list_malam})

if __name__ == '__main__':
    app.run(debug=True)