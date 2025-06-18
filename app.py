import torch
import pickle
import numpy as np
import streamlit as st
from azstemmer import AzStemmer
from scipy.sparse import hstack
import torch.nn.functional as F
from catboost import CatBoostClassifier
from functions import clean_text, add_spacing, remove_stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.classes.__path__ = []

# azstemmer kitabxanasından bir obyekt yarat
stemmer = AzStemmer(keyboard="az")

# catboost vectorizerlerini yüklə
with open('./models/catboost/title_vectorizer.pkl', 'rb') as f:
    title_vectorizer = pickle.load(f)

with open('./models/catboost/text_vectorizer.pkl', 'rb') as f:
    text_vectorizer = pickle.load(f)

# catboost modelini yüklə
catboost_model = CatBoostClassifier()
catboost_model.load_model('./models/catboost/model.cbm')

# xlm-roberta modeli üçün id2label
label_map = {0: 'təhsil', 1: 'şou-biznes', 2: 'yazarlar', 3: 'siyasət', 4: 'region', 5: 'media',
             6: 'hadisə', 7: 'i̇dman', 8: 'texnologiya', 9: 'yaşam', 10: 'mədəniyyət', 11: 'sosial',
             12: 'astrologiya', 13: 'müsahibə', 14: 'i̇qtisadiyyat', 15: 'other'}

# xlm-roberta-base modelini və vectorizerini yüklə 
model_path = "./models/xlm-roberta"
tokenizer = AutoTokenizer.from_pretrained(model_path)
xlm_model = AutoModelForSequenceClassification.from_pretrained(model_path)

# istifadəçi interfeysi
st.title("📰 Xəbər Sinifləndirmə")

model_choice = st.radio(
    "🤖 Model seçin",
    ["CatBoost", "XLM-RoBERTa"],
    horizontal=True
)

title_input = st.text_area("🔖 Xəbər başlığı", height=75)
text_input = st.text_area("🧾 Xəbər mətni", height=150)

# istifadəçiyə modeli seçmək imkanı verilir və seçilən modelə uyğun prediction qaytarılır
if st.button("Təxmin et 🤔"):
    if not title_input.strip() or not text_input.strip():
        st.warning("Zəhmət olmasa xəbər başlığını və mətnini daxil edin")
    elif len(title_input.strip().split()) < 3:
        st.warning("Xəbər başlığı ən az 3 sözdən ibarət olmalıdır!")
    elif len(text_input.strip().split()) < 20:
        st.warning("Xəbər mətni ən az 20 sözdən ibarət olmalıdır!")
    else:
        if model_choice == "CatBoost":
            preprocessed_title = remove_stopwords(add_spacing(stemmer.stem(clean_text(title_input))))
            preprocessed_text = remove_stopwords(add_spacing(stemmer.stem(clean_text(text_input))))

            title_vec = title_vectorizer.transform([preprocessed_title])
            text_vec = text_vectorizer.transform([preprocessed_text])
            combined_vec = hstack([title_vec, text_vec])

            pred_class = catboost_model.predict(combined_vec)[0]
            pred_proba = catboost_model.predict_proba(combined_vec)[0]
            confidence = np.max(pred_proba) * 100

            st.success(f"**Təxmin olunan kateqoriya:** {str(pred_class[0]).capitalize()}")
            st.success(f"{confidence:.2f}% əminəm 😊")
        
        else: 
            combined_text = f"{title_input} - {text_input}"
            preprocessed_text = clean_text(combined_text)
            
            xlm_model.eval()
            
            device = torch.device('cpu')
            xlm_model = xlm_model.to(device)
            
            inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = xlm_model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
            
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item() * 100
            pred_label = label_map[pred_id]
            
            st.success(f"**Təxmin olunan kateqoriya:** {pred_label.capitalize()}")
            st.success(f"{confidence:.2f}% əminəm 😊")