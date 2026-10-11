"""
Advanced Fashion Bot with Multiple Models and API Integration
Integrates:
1. TF-IDF + Logistic Regression
2. Word2Vec + MLP
3. BERT-based classification
4. Advanced NLP (SpaCy NER, Keyphrases, Relationships)
5. External API Integration (Hugging Face)
6. Internet Data Fetching (Google, Wikipedia, Web Scraping)
"""

import json
import pickle
import numpy as np
import random
import os
from gensim.models import Word2Vec
from nlp_utils import preprocess_text
from advanced_nlp import analyze_text_comprehensive, get_ner
from internet_data import fetch_internet_data
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AdvancedLuxuryBot:
    """
    Enhanced Fashion Bot with multiple models and API integration.
    """
    
    def __init__(self, model_type='bert'):
        """
        Initialize bot with specified model.
        
        Args:
            model_type: 'tfidf', 'word2vec', 'bert', or 'api'
        """
        self.model_type = model_type
        self.intents = None
        self.classes = None
        
        # Load intents
        with open('intents.json', 'r') as f:
            self.intents = json.load(f)
        
        # Load classes
        with open('classes.pkl', 'rb') as f:
            self.classes = pickle.load(f)
        
        # Initialize selected model
        if model_type == 'tfidf':
            self._init_tfidf_model()
        elif model_type == 'word2vec':
            self._init_word2vec_model()
        elif model_type == 'bert':
            self._init_bert_model()
        elif model_type == 'api':
            self._init_api_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"✓ Advanced Luxury Bot initialized with {model_type.upper()} model")
    
    def _init_tfidf_model(self):
        """Initialize TF-IDF + Logistic Regression model."""
        try:
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open('tfidf_model.pkl', 'rb') as f:
                self.classifier = pickle.load(f)
            print("✓ TF-IDF model loaded")
        except Exception as e:
            print(f"Error loading TF-IDF model: {e}")
            print("Please run 'python train_advanced.py' first")
            raise
    
    def _init_word2vec_model(self):
        """Initialize Word2Vec + MLP model."""
        try:
            self.word2vec = Word2Vec.load("word2vec.model")
            with open('model.pkl', 'rb') as f:
                self.classifier = pickle.load(f)
            print("✓ Word2Vec model loaded")
        except Exception as e:
            print(f"Error loading Word2Vec model: {e}")
            print("Please run 'python train_advanced.py' first")
            raise
    
    def _init_bert_model(self):
        """Initialize BERT model."""
        try:
            from transformers import BertTokenizer, BertForSequenceClassification
            import torch
            
            self.tokenizer = BertTokenizer.from_pretrained('./bert_intent_model')
            self.model = BertForSequenceClassification.from_pretrained('./bert_intent_model')
            self.model.eval()
            
            with open('label_mappings.pkl', 'rb') as f:
                mappings = pickle.load(f)
                self.label2id = mappings['label2id']
                self.id2label = mappings['id2label']
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            print(f"✓ BERT model loaded (device: {self.device})")
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            print("Please run 'python train_advanced.py' first")
            raise
    
    def _init_api_model(self):
        """Initialize Hugging Face API connection."""
        self.api_key = os.getenv('HUGGINGFACE_API_KEY')
        if not self.api_key:
            print("Warning: HUGGINGFACE_API_KEY not found in environment")
            print("Add it to .env file or set as environment variable")
            print("Get your key from: https://huggingface.co/settings/tokens")
        
        self.api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
        print("✓ Hugging Face API initialized")
    
    def predict_intent(self, sentence):
        """Predict intent using the selected model."""
        if self.model_type == 'tfidf':
            return self._predict_tfidf(sentence)
        elif self.model_type == 'word2vec':
            return self._predict_word2vec(sentence)
        elif self.model_type == 'bert':
            return self._predict_bert(sentence)
        elif self.model_type == 'api':
            return self._predict_api(sentence)
    
    def _predict_tfidf(self, sentence):
        """Predict using TF-IDF model."""
        vector = self.vectorizer.transform([sentence])
        prediction = self.classifier.predict(vector)
        probability = self.classifier.predict_proba(vector).max()
        return prediction[0], probability
    
    def _predict_word2vec(self, sentence):
        """Predict using Word2Vec model."""
        tokens = preprocess_text(sentence)
        word_vectors = []
        
        for token in tokens:
            if token in self.word2vec.wv:
                word_vectors.append(self.word2vec.wv[token])
        
        if len(word_vectors) == 0:
            return "unknown", 0.0
        
        vector = np.mean(word_vectors, axis=0).reshape(1, -1)
        prediction = self.classifier.predict(vector)
        
        try:
            probability = self.classifier.predict_proba(vector).max()
        except:
            probability = 0.5
        
        return prediction[0], probability
    
    def _predict_bert(self, sentence):
        """Predict using BERT model."""
        import torch
        
        encoding = self.tokenizer(
            sentence,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_class = torch.max(predictions, dim=1)
        
        intent = self.id2label[predicted_class.item()]
        probability = confidence.item()
        
        return intent, probability
    
    def _predict_api(self, sentence):
        """
        Predict using Hugging Face Inference API.
        
        This demonstrates real-world API-based NLP deployment:
        - No local model needed
        - Scalable cloud infrastructure
        - Access to state-of-the-art models
        - Pay-per-use pricing
        """
        if not self.api_key:
            return "greeting", 0.5  # Fallback
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Use zero-shot classification
        payload = {
            "inputs": sentence,
            "parameters": {"candidate_labels": self.classes}
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                intent = result['labels'][0]
                probability = result['scores'][0]
                return intent, probability
            else:
                print(f"API Error: {response.status_code}")
                return "greeting", 0.5
        
        except Exception as e:
            print(f"API Request failed: {e}")
            return "greeting", 0.5
    
    def get_response(self, sentence, use_advanced_nlp=True, use_internet=False):
        """
        Get response with advanced NLP analysis and internet data.
        
        Args:
            sentence: User input
            use_advanced_nlp: Whether to use advanced NLP features
            use_internet: Whether to fetch real-time data from internet
        
        Returns:
            Dictionary with response and analysis
        """
        # Predict intent
        intent_tag, confidence = self.predict_intent(sentence)
        
        # Advanced NLP Analysis (Unit 4)
        analysis = None
        if use_advanced_nlp:
            try:
                analysis = analyze_text_comprehensive(sentence)
            except Exception as e:
                print(f"Advanced NLP error: {e}")
                analysis = None
        
        # Internet Data Fetching
        internet_data = None
        if use_internet:
            try:
                internet_data = fetch_internet_data(sentence)
            except Exception as e:
                print(f"Internet fetch error: {e}")
                internet_data = None
        
        # Get base response
        response = self._get_base_response(intent_tag)
        
        # Enhance response with NLP insights
        if analysis:
            response = self._enhance_response(response, analysis, intent_tag)
        
        # Enhance with internet data
        if internet_data:
            response = self._enhance_with_internet_data(response, internet_data, sentence)
        
        return {
            'response': response,
            'intent': intent_tag,
            'confidence': confidence,
            'model': self.model_type,
            'analysis': analysis,
            'internet_data': internet_data
        }
    
    def _get_base_response(self, intent_tag):
        """Get base response for intent."""
        for intent in self.intents['intents']:
            if intent['tag'] == intent_tag:
                return random.choice(intent['responses'])
        
        return "I apologize, but I'm not sure I understand. Could you rephrase using luxury fashion terms?"
    
    def _enhance_response(self, response, analysis, intent_tag):
        """Enhance response using NLP analysis."""
        
        # Add brand acknowledgment
        if 'fashion_entities' in analysis:
            brands = analysis['fashion_entities'].get('brands', [])
            products = analysis['fashion_entities'].get('products', [])
            
            if brands and intent_tag not in ['greeting', 'goodbye']:
                brand_text = ', '.join(brands)
                response = f"Regarding {brand_text}: {response}"
            
            if products and intent_tag == 'product_inquiry':
                product_text = ', '.join(products)
                response = f"{response} I can help you find the perfect {product_text}."
        
        # Add sentiment-based personalization
        if 'sentiment' in analysis:
            sentiment = analysis['sentiment']['sentiment']
            if sentiment == 'negative':
                response = "I understand your concern. " + response
            elif sentiment == 'positive':
                response += " I'm delighted to assist you!"
        
        return response
    
    def _enhance_with_internet_data(self, response, internet_data, sentence):
        """Enhance response with real-time internet data."""
        try:
            if not internet_data or not internet_data.get('data'):
                return response
            
            data = internet_data['data'][0] if internet_data['data'] else None
            if not data:
                return response
            
            # Extract search results
            search_results = data.get('search_results', [])
            
            if search_results:
                # Add internet data citation
                top_result = search_results[0]
                response += f"\n\n📱 Real-time Info: {top_result.get('summary', top_result.get('title', ''))[:200]}..."
                
                if top_result.get('url'):
                    response += f"\n🔗 Source: {top_result['source']}"
            
            # Add specific information based on data type
            if 'brand' in data:
                brand_info = data.get('local_info', {})
                if brand_info and 'name' in brand_info:
                    response += f"\n\n{brand_info['name']} - Founded {brand_info.get('founded', 'N/A')} in {brand_info.get('country', 'N/A')}"
            
            if 'estimated_price_range' in data:
                response += f"\n💰 Price Range: {data['estimated_price_range']}"
            
            if 'trend_keywords' in data:
                trends = data['trend_keywords'][:3]
                response += f"\n🔥 Trending: {', '.join(trends)}"
        
        except Exception as e:
            print(f"Error enhancing with internet data: {e}")
        
        return response
    
    def compare_models(self, sentence):
        """
        Compare predictions from all available models.
        Useful for debugging and analysis.
        """
        results = {}
        
        original_model = self.model_type
        
        for model_type in ['tfidf', 'word2vec', 'bert']:
            try:
                self.model_type = model_type
                if model_type == 'tfidf':
                    self._init_tfidf_model()
                elif model_type == 'word2vec':
                    self._init_word2vec_model()
                elif model_type == 'bert':
                    self._init_bert_model()
                
                intent, confidence = self.predict_intent(sentence)
                results[model_type] = {
                    'intent': intent,
                    'confidence': confidence
                }
            except Exception as e:
                results[model_type] = {'error': str(e)}
        
        # Restore original model
        self.model_type = original_model
        
        return results

# ==================== API Integration Demo ====================

class HuggingFaceAPIDemo:
    """
    Demonstrates various Hugging Face API capabilities.
    
    API-based deployment advantages:
    1. No local model storage/training needed
    2. Instant access to latest models
    3. Automatic scaling
    4. Multiple models available
    5. Cost-effective for low-volume applications
    """
    
    def __init__(self):
        self.api_key = os.getenv('HUGGINGFACE_API_KEY')
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
    
    def sentiment_analysis(self, text):
        """Sentiment analysis via API."""
        api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
        response = requests.post(api_url, headers=self.headers, json={"inputs": text})
        return response.json() if response.status_code == 200 else None
    
    def text_generation(self, prompt):
        """Text generation via API."""
        api_url = "https://api-inference.huggingface.co/models/gpt2"
        response = requests.post(api_url, headers=self.headers, json={"inputs": prompt})
        return response.json() if response.status_code == 200 else None
    
    def question_answering(self, question, context):
        """Question answering via API."""
        api_url = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
        payload = {"inputs": {"question": question, "context": context}}
        response = requests.post(api_url, headers=self.headers, json=payload)
        return response.json() if response.status_code == 200 else None

# ==================== Initialize Bot ====================

# Default bot instance (can be changed in app.py)
chatbot = None

def initialize_bot(model_type='bert'):
    """Initialize the bot with specified model."""
    global chatbot
    chatbot = AdvancedLuxuryBot(model_type=model_type)
    return chatbot

def get_bot_response(user_input, use_advanced_nlp=True, use_internet=False):
    """Get bot response (wrapper for Flask app)."""
    global chatbot
    if chatbot is None:
        chatbot = initialize_bot('word2vec')
    
    return chatbot.get_response(user_input, use_advanced_nlp, use_internet)

# ==================== Testing ====================

if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED FASHION BOT TEST")
    print("=" * 60 + "\n")
    
    # Test with different models
    test_sentence = "I'm looking for a beautiful Gucci handbag"
    
    print(f"Test Input: '{test_sentence}'\n")
    
    for model_type in ['tfidf', 'word2vec', 'bert']:
        print(f"\n{model_type.upper()} Model:")
        print("-" * 40)
        
        try:
            bot = AdvancedLuxuryBot(model_type=model_type)
            result = bot.get_response(test_sentence)
            
            print(f"Intent: {result['intent']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Response: {result['response']}")
            
            if result['analysis']:
                print(f"\nNLP Analysis:")
                print(f"  Brands: {result['analysis']['fashion_entities']['brands']}")
                print(f"  Products: {result['analysis']['fashion_entities']['products']}")
                print(f"  Sentiment: {result['analysis']['sentiment']['sentiment']}")
        
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 60)
