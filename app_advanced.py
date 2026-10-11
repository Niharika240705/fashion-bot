"""
Advanced Flask Application for Fashion Bot
Features:
- Multiple model selection (TF-IDF, Word2Vec, BERT, API)
- Advanced NLP analysis display
- Model comparison endpoint
- API integration
"""

from flask import Flask, render_template, request, jsonify
from chatbot_advanced import initialize_bot, get_bot_response, HuggingFaceAPIDemo
import os

app = Flask(__name__)

# Initialize bot with default model
current_model = 'bert'
try:
    bot = initialize_bot(current_model)
    print(f"✓ Bot initialized with {current_model} model")
except Exception as e:
    print(f"Error initializing bot: {e}")
    print("Falling back to word2vec model...")
    try:
        current_model = 'word2vec'
        bot = initialize_bot(current_model)
    except:
        print("Error: Could not initialize any model. Please run training first.")
        bot = None

@app.route("/")
def home():
    """Render main page."""
    return render_template("index.html", current_model=current_model)

@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.
    Returns response with advanced NLP analysis and optional internet data.
    BULLETPROOF VERSION - Always returns a response, never crashes.
    """
    try:
        user_input = request.json.get("message", "").strip()
        use_advanced_nlp = request.json.get("use_advanced_nlp", True)
        use_internet = request.json.get("use_internet", True)  # Internet search ALWAYS ON by default
        
        if not user_input:
            return jsonify({"response": "Please say something."})
        
        if bot is None:
            return jsonify({
                "response": "Welcome to Luxury Fashion Bot! Ask me about designer brands, fashion trends, or luxury products.",
                "intent": "greeting",
                "confidence": "1.00",
                "model": "FALLBACK",
                "note": "Bot initializing..."
            })
        
        try:
            result = get_bot_response(user_input, use_advanced_nlp, use_internet)
            
            # Format response for frontend
            response_data = {
                "response": result['response'],
                "intent": result['intent'],
                "confidence": f"{result['confidence']:.2%}",
                "model": result['model'].upper(),
                "internet_enabled": use_internet
            }
            
            # Add NLP analysis if available
            if result.get('analysis'):
                try:
                    analysis = result['analysis']
                    response_data['analysis'] = {
                        'brands': analysis.get('fashion_entities', {}).get('brands', []),
                        'products': analysis.get('fashion_entities', {}).get('products', []),
                        'sentiment': analysis.get('sentiment', {}).get('sentiment', 'neutral'),
                        'keyphrases': analysis.get('keyphrases', {}).get('keybert', [])[:3]
                    }
                except Exception as e:
                    print(f"Analysis error: {e}")
            
            # Add internet data if available
            if result.get('internet_data'):
                try:
                    internet_data = result['internet_data']
                    if internet_data.get('data'):
                        data = internet_data['data'][0]
                        search_results = data.get('search_results', [])
                        
                        response_data['internet_data'] = {
                            'source': search_results[0].get('source', 'Web') if search_results else 'Web',
                            'results_count': len(search_results),
                            'search_results': search_results[:3],  # Include top 3 results
                            'has_data': len(search_results) > 0
                        }
                        print(f"✓ Internet data added: {len(search_results)} results from {response_data['internet_data']['source']}")
                except Exception as e:
                    print(f"Internet data error: {e}")
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Error in get_bot_response: {e}")
            # Fallback response
            return jsonify({
                "response": "That's a fascinating question about fashion! I apologize, but I'm having trouble processing that right now. Please try rephrasing your question.",
                "intent": "unknown",
                "confidence": "0.50",
                "model": "FALLBACK"
            })
    
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        # Ultimate fallback - always return something
        return jsonify({
            "response": "Thank you for your interest in luxury fashion. Could you please rephrase your question?",
            "intent": "general",
            "confidence": "0.50",
            "model": "FALLBACK"
        })

@app.route("/switch_model", methods=["POST"])
def switch_model():
    """
    Switch between different models.
    Allows comparison of TF-IDF, Word2Vec, BERT, and API models.
    """
    global bot, current_model
    
    model_type = request.json.get("model_type")
    
    if model_type not in ['tfidf', 'word2vec', 'bert', 'api']:
        return jsonify({
            "success": False,
            "message": "Invalid model type"
        })
    
    try:
        bot = initialize_bot(model_type)
        current_model = model_type
        
        return jsonify({
            "success": True,
            "message": f"Switched to {model_type.upper()} model",
            "model": model_type
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error switching model: {str(e)}"
        })

@app.route("/compare_models", methods=["POST"])
def compare_models():
    """
    Compare predictions from all available models.
    Useful for demonstration and analysis.
    """
    user_input = request.json.get("message")
    
    if not user_input:
        return jsonify({"error": "No message provided"})
    
    try:
        results = bot.compare_models(user_input)
        
        # Format results
        comparison = []
        for model_name, result in results.items():
            if 'error' not in result:
                comparison.append({
                    'model': model_name.upper(),
                    'intent': result['intent'],
                    'confidence': f"{result['confidence']:.2%}"
                })
            else:
                comparison.append({
                    'model': model_name.upper(),
                    'error': result['error']
                })
        
        return jsonify({
            "success": True,
            "comparisons": comparison
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route("/api_demo", methods=["POST"])
def api_demo():
    """
    Demonstrate Hugging Face API capabilities.
    Shows various NLP tasks via API.
    """
    task = request.json.get("task")
    text = request.json.get("text")
    
    api_demo = HuggingFaceAPIDemo()
    
    try:
        if task == "sentiment":
            result = api_demo.sentiment_analysis(text)
        elif task == "generation":
            result = api_demo.text_generation(text)
        elif task == "qa":
            context = request.json.get("context", "")
            result = api_demo.question_answering(text, context)
        else:
            return jsonify({"error": "Unknown task"})
        
        return jsonify({
            "success": True,
            "result": result
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route("/internet_search", methods=["POST"])
def internet_search():
    """
    Search the internet for fashion information.
    Uses Google, Wikipedia, and web scraping.
    """
    query = request.json.get("query")
    
    if not query:
        return jsonify({"error": "Query required"})
    
    try:
        from internet_data import fetch_internet_data
        
        result = fetch_internet_data(query)
        
        # Format results
        formatted_data = {
            'query': result['query'],
            'timestamp': result['timestamp'],
            'results': []
        }
        
        if result.get('data'):
            data = result['data'][0]
            search_results = data.get('search_results', [])
            
            for res in search_results[:5]:  # Top 5 results
                formatted_data['results'].append({
                    'title': res.get('title', ''),
                    'summary': res.get('summary', ''),
                    'url': res.get('url', ''),
                    'source': res.get('source', 'Web')
                })
        
        return jsonify({
            "success": True,
            "data": formatted_data
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "current_model": current_model,
        "bot_initialized": bot is not None
    })

@app.route("/info")
def info():
    """
    Information about the NLP concepts and models.
    Educational endpoint for Units 3, 4, 5.
    """
    info_data = {
        "units": {
            "unit_3": {
                "title": "Neural Embeddings & Deep Learning",
                "concepts": [
                    "TF-IDF: Term Frequency-Inverse Document Frequency",
                    "Word2Vec: Distributed word representations",
                    "Neural Networks: MLP with backpropagation",
                    "Model Comparison: Baseline vs Deep Learning"
                ],
                "models": ["TF-IDF + Logistic Regression", "Word2Vec + MLP"]
            },
            "unit_4": {
                "title": "Information Extraction",
                "concepts": [
                    "Named Entity Recognition (SpaCy)",
                    "Keyphrase Extraction (KeyBERT, YAKE)",
                    "Relationship Extraction",
                    "Dependency Parsing"
                ],
                "features": ["Fashion brand detection", "Product entity extraction", "Sentiment analysis"]
            },
            "unit_5": {
                "title": "Transformers & Advanced Models",
                "concepts": [
                    "BERT: Bidirectional Encoder Representations",
                    "Self-Attention Mechanism",
                    "Multi-Head Attention",
                    "Fine-tuning Pre-trained Models",
                    "Transfer Learning"
                ],
                "models": ["BERT-based Intent Classification", "Hugging Face API Integration"]
            }
        },
        "api_integration": {
            "provider": "Hugging Face Inference API",
            "benefits": [
                "No local model storage needed",
                "Access to state-of-the-art models",
                "Automatic scaling",
                "Cost-effective for production"
            ],
            "endpoints": ["Zero-shot classification", "Sentiment analysis", "Text generation"]
        }
    }
    
    return jsonify(info_data)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ADVANCED FASHION BOT - FLASK APPLICATION")
    print("=" * 60)
    print(f"Current Model: {current_model.upper()}")
    print("\nEndpoints:")
    print("  • /          - Main chat interface")
    print("  • /chat      - Chat API")
    print("  • /switch_model - Switch between models")
    print("  • /compare_models - Compare all models")
    print("  • /api_demo  - Hugging Face API demo")
    print("  • /health    - Health check")
    print("  • /info      - NLP concepts information")
    print("=" * 60 + "\n")
    
    app.run(debug=False, port=5003)
