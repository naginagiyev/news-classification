# Classification Model for Azerbaijan News 

This project implements a news classification system for Azerbaijani text using two different models: CatBoost and XLM-RoBERTa. The system can classify news articles into various categories such as politics, sports, technology, and more.

## Project Structure

```
.
├── app.py                  # Streamlit web interface
├── api.py                  # FastAPI server
├── functions.py            # Utility functions for text processing
├── preprocessing.ipynb     # Data preprocessing notebook
├── catboosttrain.ipynb     # CatBoost model training notebook
├── xlmtraining.ipynb       # XLM-RoBERTa model training notebook
├── models/                 # Directory containing trained models
│   ├── catboost/           # CatBoost model files
│   └── xlm-roberta/        # XLM-RoBERTa model files
├── data/                   # Data directory
├── results/                # Training results and evaluations
└── files/                  # Additional project files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/naginagiyev/news-classification.git
cd news-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Python version of the environment for that task is 3.10.16

## Running the Application

Before running the application, you should train XLM-Roberta-Base model and let it save the model and its tokenizers into directory. The model itself is too large to push to Github. That's why, firstly you need to train it or use a pre-trained model and place it on the folder.

### Web Interface (Streamlit)
There exist a user interface where you can select and test both of the trained models.In order to run the Streamlit web interface, run following command:
```bash
streamlit run app.py
```
This will start a local web server where you can:
- Choose between CatBoost and XLM-RoBERTa models
- Input news title and text
- Get classification predictions with confidence scores

### API Server (FastAPI)
In order to run the FastAPI server, run following command on your terminal:
```bash
uvicorn api:app --reload
```
The API will be available at `http://localhost:8000`

API Endpoints:
- `POST /predict`: Get predictions for news text
  - Request body: `{"title": "news title", "text": "news text", "model": "catboost" or "xlm"}`
  - Response: `{"category": "predicted category", "confidence": confidence_score}`

## Model Training

Before the train models, you need to open preprocessing.ipynb and run all cells with order. As Github does not accept large files like the dataset we'll use in this training, you need to preprocess and save it before the running training codes.

### CatBoost Model Training

1. Open and run the `catboosttrain.ipynb` notebook
2. The notebook will:
   - Load and preprocess the data
   - Train the CatBoost model
   - Save the model and vectorizers to the `./models/catboost/` directory

### XLM-RoBERTa Model Training
1. Open and run the `xlmtraining.ipynb` notebook
2. The notebook will:
   - Load and preprocess the data
   - Fine-tune the XLM-RoBERTa model
   - Save the model to the `./models/xlm-roberta/` directory

**NOTE:** The XLM-Roberta model is trained on Google Colab in more than 3 hours overall. It is not recommended to run such a large model on a local computer, as small notebooks are lack of GPU and it will probably cause errors such as OutOfMemory error.