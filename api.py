import torch
import pickle
import numpy as np
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
from azstemmer import AzStemmer
from scipy.sparse import hstack
import torch.nn.functional as F
from catboost import CatBoostClassifier
from functions import clean_text, add_spacing, remove_stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NewsInput(BaseModel):
    model: Literal["xlm", "catboost"]
    title: str
    text: str

app = FastAPI()
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

# /predict endpoint
@app.post("/predict")
def predict(input_data: NewsInput):
    cleaned_title = clean_text(input_data.title)
    cleaned_text = clean_text(input_data.text)

    # daha dəqiq nəticə üçün istifadəçidən train olunan tekstlərin uzunluğuna uyğun inputlar istə
    if len(cleaned_title.split()) < 3:
        return {"warning":"News title should be at least 3 words!"}
    
    elif len(cleaned_text.split()) < 20:
        return {"warning":"News text should be at least 20 words!"}

    else:
        if input_data.model == "catboost":
            preprocessed_title = remove_stopwords(add_spacing(stemmer.stem(cleaned_title)))
            preprocessed_text = remove_stopwords(add_spacing(stemmer.stem(cleaned_text)))
            title_vec = title_vectorizer.transform([preprocessed_title])
            text_vec = text_vectorizer.transform([preprocessed_text])
            combined_vec = hstack([title_vec, text_vec])

            pred_class = str(catboost_model.predict(combined_vec)[0][0]).capitalize()
            pred_proba = catboost_model.predict_proba(combined_vec)[0]
            confidence = float(np.max(pred_proba) * 100)

            return {"prediction": pred_class, "confidence": round(confidence, 2)}

        else:
            preprocessed_text = f"{cleaned_title} - {cleaned_text}"
            xlm_model.eval()
            
            device = torch.device('cpu')
            xlm_model.to(device)
            
            inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = xlm_model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
            
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item() * 100
            pred_label = label_map[pred_id]

            return {"prediction": pred_label.capitalize(), "confidence": round(confidence, 2)}