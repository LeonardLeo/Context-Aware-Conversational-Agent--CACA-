# Context-Aware Conversational Agent (CACA) - Full Implementation

import pandas as pd
import re
import nltk
import spacy
from nltk.tokenize import word_tokenize
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, util

# Load dataset (DailyDialog)
def load_data():
    url = "https://raw.githubusercontent.com/jfainberg/dailydialog/master/dailydialog.csv"
    df = pd.read_csv(url)
    return df[['utterance']]

# Preprocessing function
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    return ' '.join(tokens)

# Sentiment Analysis Setup
sentiment_analyzer = pipeline('sentiment-analysis')

def analyze_sentiment(text):
    return sentiment_analyzer(text)[0]['label']

# Named Entity Recognition (NER)
ner_pipeline = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')

def extract_entities(text):
    return ner_pipeline(text)

# Load Chatbot Model (Hugging Face Transformer-based)
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(user_input):
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

# Intent Recognition using Sentence Transformers
intent_model = SentenceTransformer("all-MiniLM-L6-v2")
intent_labels = ["greeting", "farewell", "question", "request", "complaint"]
intent_embeddings = intent_model.encode(intent_labels)

def get_intent(text):
    text_embedding = intent_model.encode(text)
    similarities = util.pytorch_cos_sim(text_embedding, intent_embeddings)[0]
    intent_idx = similarities.argmax().item()
    return intent_labels[intent_idx]

# FastAPI for Deployment
app = FastAPI()

@app.post("/chat")
def chatbot_api(user_input: str):
    sentiment = analyze_sentiment(user_input)
    intent = get_intent(user_input)
    response = generate_response(user_input)
    return {"response": response, "sentiment": sentiment, "intent": intent}

# Example Usage
if __name__ == "__main__":
    df = load_data()
    df['processed_text'] = df['utterance'].apply(preprocess_text)
    df['sentiment'] = df['processed_text'].apply(analyze_sentiment)
    df['entities'] = df['processed_text'].apply(extract_entities)
    print(df.head())
