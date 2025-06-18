import re 

# bu funksiya xəbər başlıqlarını və tekstlərini lazımsız simvollardan təmizləmək üçündür
def clean_text(text):
    return re.sub(r"[^\w\d\s\.\,\?\!\:\-]|_", "", text, flags=re.UNICODE)

# stemming prosesindən sonra bəzi simvollar, hərflər və rəqəmlər bitişik şəkildə gəlir və bu funksiya onların arasına boşluq əlavə etmək üçündür
def add_spacing(text):
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

    text = re.sub(r'([a-zA-Z])([^\w\s])', r'\1 \2', text)
    text = re.sub(r'([^\w\s])([a-zA-Z])', r'\1 \2', text)

    text = re.sub(r'(\d)([^\w\s])', r'\1 \2', text)
    text = re.sub(r'([^\w\s])(\d)', r'\1 \2', text)
    return text

# ./files/stopwords.txt faylında olan stop wordləri mətnlərdən silmək üçündür
def remove_stopwords(text):
    with open("./files/stopwords.txt", "r") as file:
        stopwords = [word.strip() for word in file.readlines()]

    pattern = r'\b(?:' + '|'.join(re.escape(word) for word in stopwords) + r')\b'
    cleaned = re.sub(pattern, '', text)
    return re.sub(r'\s{2,}', ' ', cleaned).strip()