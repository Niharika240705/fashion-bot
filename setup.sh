#!/bin/bash

# Setup Script for Advanced Fashion Bot
# Automates installation and setup process

echo "=========================================="
echo "Advanced Fashion Bot - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source venv/bin/activate
else
    source venv/Scripts/activate
fi
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Download SpaCy model
echo "Downloading SpaCy model..."
python -m spacy download en_core_web_sm
echo "✓ SpaCy model downloaded"
echo ""

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
echo "✓ NLTK data downloaded"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "✓ .env file created"
    echo "⚠️  Remember to add your Hugging Face API key to .env"
else
    echo "✓ .env file already exists"
fi
echo ""

# Test advanced NLP
echo "Testing advanced NLP utilities..."
python advanced_nlp.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Advanced NLP utilities working"
else
    echo "⚠️  Advanced NLP test had some warnings (this is usually okay)"
fi
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Train the models:"
echo "   python train_advanced.py"
echo ""
echo "2. Run the chatbot:"
echo "   python app_advanced.py"
echo ""
echo "3. Open browser:"
echo "   http://localhost:5003"
echo ""
echo "Optional: Add Hugging Face API key to .env for API model"
echo "Get key from: https://huggingface.co/settings/tokens"
echo ""
echo "=========================================="
