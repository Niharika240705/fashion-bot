import json
import pickle
import numpy as np
import random
from gensim.models import Word2Vec
from nlp_utils import preprocess_text, get_named_entities, extract_noun_phrases

class LuxuryBot:
    def __init__(self):
        self.word2vec = None
        self.classifier = None
        self.classes = None
        self.intents = None
        self.load_models()

    def load_models(self):
        try:
            self.word2vec = Word2Vec.load("word2vec.model")
            with open('model.pkl', 'rb') as f:
                self.classifier = pickle.load(f)
            with open('classes.pkl', 'rb') as f:
                self.classes = pickle.load(f)
            with open('intents.json', 'r') as f:
                self.intents = json.load(f)
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please ensure you have run 'python train.py' first.")

    def get_sentence_vector(self, sentence):
        tokens = preprocess_text(sentence)
        word_vectors = []
        for token in tokens:
            if token in self.word2vec.wv:
                word_vectors.append(self.word2vec.wv[token])
        
        if len(word_vectors) == 0:
            return np.zeros(self.word2vec.vector_size)
        return np.mean(word_vectors, axis=0)

    def predict_intent(self, sentence):
        if not self.classifier:
            return "unknown"
            
        vector = self.get_sentence_vector(sentence)
        # Vector needs to be reshaped for single sample prediction
        vector = vector.reshape(1, -1)
        prediction = self.classifier.predict(vector)
        return prediction[0]

    def get_response(self, sentence):
        intent_tag = self.predict_intent(sentence)
        
        # Advanced NLP Analysis
        entities = get_named_entities(sentence)
        product_phrases = extract_noun_phrases(sentence)
        
        # Log analysis for demonstration (or debugging)
        if entities:
            print(f"[Debug] Detected Entities: {entities}")
        if product_phrases:
            print(f"[Debug] Extracted Phrases: {product_phrases}")
            
        # Select response based on intent
        response = "I'm not sure how to help with that."
        
        for intent in self.intents['intents']:
            if intent['tag'] == intent_tag:
                response = random.choice(intent['responses'])
                break
        
        # Personalization using NER
        # If we found a PERSON or ORG entity (likely a Brand), we can acknowledge it
        brands_found = [e[0] for e in entities if e[1] in ['PERSON', 'ORGANIZATION', 'GPE']]
        if brands_found and intent_tag not in ['greeting', 'goodbye']:
            # Simple enhancement: Prepend an acknowledgement
            response = f"Regarding {brands_found[0]}: {response}"
            
        return response
        
        return "I apologize, but I'm not sure I understand. Could you rephrase using luxury fashion terms?"

chatbot = LuxuryBot()

def get_bot_response(user_input):
    return chatbot.get_response(user_input)
