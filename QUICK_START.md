# 🚀 Quick Start Guide - Advanced Fashion Bot

## Installation (5 minutes)

### Option 1: Automated Setup (Recommended)

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup script
./setup.sh
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download SpaCy model
python -m spacy download en_core_web_sm

# 4. Create environment file (optional)
cp .env.template .env
```

## Training Models (10-20 minutes)

```bash
# Train all three models: TF-IDF, Word2Vec, BERT
python train_advanced.py

# Expected output:
# ✓ TF-IDF model: ~85% accuracy
# ✓ Word2Vec model: ~88% accuracy  
# ✓ BERT model: ~92% accuracy
# ✓ Model comparison saved
```

**Note:** BERT training takes 10-15 minutes. For faster testing, you can train only TF-IDF and Word2Vec first.

## Running the Application

### Advanced Interface (Recommended)

```bash
python app_advanced.py
```

Open browser: `http://localhost:5003`

### Original Interface

```bash
python app.py
```

## Testing Features

### 1. Test Model Switching
- Click different model buttons (TF-IDF, Word2Vec, BERT, API)
- Send same message with different models
- Compare accuracy and responses

### 2. Test Advanced NLP (Unit 4)
- Enable "Advanced NLP Analysis" toggle
- Send: "I'm looking for a beautiful Gucci handbag"
- View detected brands, products, sentiment, keyphrases

### 3. Test Model Comparison
- Type a message
- Click "📊 Compare Models" button
- See predictions from all models side-by-side

### 4. View NLP Concepts
- Click "ℹ️ NLP Info" button
- Browse Units 3, 4, 5 concepts
- Learn about each technology

## Example Queries

```
✅ "Show me luxury handbags"
   • Intent: product_inquiry
   • Products detected: handbag
   • Model: BERT

✅ "I'm looking for a Gucci dress for a wedding"
   • Brands: Gucci
   • Products: dress
   • Keyphrases: gucci dress, wedding
   • Sentiment: positive

✅ "Hello, I need fashion advice"
   • Intent: greeting + advice
   • Sentiment: neutral
```

## API Integration (Optional)

### Get Hugging Face API Key

1. Go to: https://huggingface.co/settings/tokens
2. Create new token (read access)
3. Add to `.env` file:
   ```
   HUGGINGFACE_API_KEY=hf_your_actual_key_here
   ```
4. Restart Flask app
5. Select "HF API" model

**Benefits:**
- No local BERT model needed
- Access to latest models
- Zero-shot classification
- Cloud-based inference

## Troubleshooting

### ❌ "No module named 'transformers'"
```bash
pip install transformers torch
```

### ❌ "SpaCy model not found"
```bash
python -m spacy download en_core_web_sm
```

### ❌ "BERT training is too slow"
```bash
# Option 1: Skip BERT for now, use TF-IDF/Word2Vec
# Option 2: Reduce epochs in train_advanced.py (line 220)
num_train_epochs=1  # Change from 3 to 1
```

### ❌ "Memory error during training"
```bash
# Train models separately:
# 1. Comment out BERT training in train_advanced.py
# 2. Train TF-IDF and Word2Vec first
# 3. Uncomment BERT and train separately
```

## Project Files Overview

### Core Files
- `train_advanced.py` - Train all 3 models
- `app_advanced.py` - Advanced Flask application
- `chatbot_advanced.py` - Multi-model chatbot
- `advanced_nlp.py` - SpaCy, KeyBERT, relationships

### Original Files
- `train.py` - Original training
- `app.py` - Original Flask app
- `chatbot.py` - Original chatbot
- `nlp_utils.py` - Basic NLP utilities

### Data & Config
- `intents.json` - Training data
- `requirements.txt` - Dependencies
- `.env` - Configuration (create from .env.template)

### Generated Models
- `tfidf_model.pkl` - TF-IDF classifier
- `word2vec.model` - Word embeddings
- `bert_intent_model/` - Fine-tuned BERT
- `model_comparison.json` - Performance metrics

## Quick Commands Reference

```bash
# Install everything
./setup.sh

# Train models
python train_advanced.py

# Test NLP features
python advanced_nlp.py

# Run chatbot (advanced)
python app_advanced.py

# Run chatbot (original)
python app.py

# Test individual model
python chatbot_advanced.py
```

## Next Steps

1. ✅ Train all models
2. ✅ Test in web interface
3. ✅ Try different queries
4. ✅ Compare model performance
5. ✅ Explore NLP analysis
6. ⭐ (Optional) Add Hugging Face API key
7. ⭐ (Optional) Customize intents.json
8. ⭐ (Optional) Add more training data

## Learning Path

### Beginner
1. Start with TF-IDF model (simple, fast)
2. Understand baseline performance
3. Try Word2Vec for semantic understanding

### Intermediate
1. Explore BERT transformer model
2. Study self-attention mechanism
3. Compare model predictions

### Advanced
1. Integrate Hugging Face API
2. Customize relationship extraction
3. Add new entity types
4. Fine-tune on larger datasets

## Resources

- 📖 Full documentation: `README_ADVANCED.md`
- 🎓 NLP concepts: Click "ℹ️ NLP Info" in app
- 🔗 Hugging Face: https://huggingface.co
- 🔗 SpaCy: https://spacy.io
- 🔗 BERT paper: https://arxiv.org/abs/1810.04805

## Support

**Common Issues:**
- Training errors → Check Python version (3.8+)
- Import errors → Reinstall requirements.txt
- Slow training → Reduce epochs or skip BERT
- API errors → Check API key in .env

**Still stuck?**
- Check error messages carefully
- Verify all dependencies installed
- Try running individual components
- Review README_ADVANCED.md for details

---

**Ready to start? Run:**

```bash
./setup.sh && python train_advanced.py && python app_advanced.py
```

**Happy Learning! 🎓**
