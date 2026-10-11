"""
Advanced Training Script for Fashion Bot
Implements multiple models for comparison:
1. TF-IDF + Logistic Regression (Baseline - Unit 3)
2. Word2Vec + MLP (Original model)
3. BERT-based Intent Classification (Transformer - Unit 5)

Includes detailed evaluation metrics and model comparison.
"""

import json
import numpy as np
import pickle
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from nlp_utils import preprocess_text
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

# ==================== Data Preparation ====================

def load_and_prepare_data():
    """Load intents and prepare training data."""
    print("=" * 60)
    print("LOADING AND PREPARING DATA")
    print("=" * 60)
    
    with open('intents.json', 'r') as f:
        intents = json.load(f)

    sentences = []
    labels = []
    classes = []
    original_sentences = []  # Keep original for TF-IDF
    
    for intent in intents['intents']:
        tag = intent['tag']
        if tag not in classes:
            classes.append(tag)
            
        for pattern in intent['patterns']:
            original_sentences.append(pattern)
            tokens = preprocess_text(pattern)
            if tokens:
                sentences.append(tokens)
                labels.append(tag)
    
    print(f"✓ Loaded {len(sentences)} training samples")
    print(f"✓ Number of intent classes: {len(classes)}")
    print(f"✓ Classes: {classes}\n")
    
    return sentences, original_sentences, labels, classes, intents

# ==================== MODEL 1: TF-IDF + Logistic Regression ====================

def train_tfidf_model(original_sentences, labels, classes):
    """
    UNIT 3: TF-IDF + Logistic Regression Baseline
    
    TF-IDF (Term Frequency-Inverse Document Frequency):
    - Captures importance of words in documents
    - TF: How often a word appears in document
    - IDF: How rare the word is across all documents
    - Formula: TF-IDF(t,d) = TF(t,d) × log(N / DF(t))
    
    Why this is a good baseline:
    - Simple, interpretable, fast
    - Works well for text classification
    - Industry standard baseline for comparison
    """
    print("=" * 60)
    print("MODEL 1: TF-IDF + LOGISTIC REGRESSION (BASELINE)")
    print("=" * 60)
    
    # Create TF-IDF vectors
    print("Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=1,
        max_df=0.8
    )
    
    X_tfidf = vectorizer.fit_transform(original_sentences)
    print(f"✓ TF-IDF matrix shape: {X_tfidf.shape}")
    
    # Split data (handle small datasets without stratification)
    min_class_samples = min([labels.count(c) for c in set(labels)])
    use_stratify = len(labels) > len(set(labels)) * 2  # Need at least 2 samples per class
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, labels, test_size=0.2, random_state=42, 
        stratify=labels if use_stratify else None
    )
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        multi_class='multinomial',
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Training completed!")
    print(f"✓ Test Accuracy: {accuracy:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(lr_model, X_tfidf, labels, cv=5)
    print(f"✓ Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('tfidf_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    
    print("✓ Model saved: tfidf_model.pkl\n")
    
    return {
        'model': lr_model,
        'vectorizer': vectorizer,
        'accuracy': accuracy,
        'cv_scores': cv_scores,
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

# ==================== MODEL 2: Word2Vec + MLP ====================

def train_word2vec_model(sentences, labels, classes):
    """
    Original Model: Word2Vec + Multi-Layer Perceptron
    
    Word2Vec:
    - Learns distributed word representations
    - Words with similar meanings have similar vectors
    - Two architectures: CBOW (predict word from context) or Skip-gram
    
    MLP (Multi-Layer Perceptron):
    - Neural network with hidden layers
    - Can learn non-linear decision boundaries
    - Backpropagation for training
    """
    print("=" * 60)
    print("MODEL 2: WORD2VEC + MLP (NEURAL EMBEDDINGS)")
    print("=" * 60)
    
    # Train Word2Vec
    print("Training Word2Vec model...")
    word2vec_model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        epochs=10
    )
    word2vec_model.save("word2vec.model")
    print("✓ Word2Vec model saved")
    
    # Convert sentences to vectors
    X = []
    y = []
    
    for i, tokens in enumerate(sentences):
        vec = []
        for word in tokens:
            if word in word2vec_model.wv:
                vec.append(word2vec_model.wv[word])
        
        if len(vec) > 0:
            X.append(np.mean(vec, axis=0))
            y.append(labels[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data (handle small datasets without stratification)
    use_stratify = len(y) > len(np.unique(y)) * 2  # Need at least 2 samples per class
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=y if use_stratify else None
    )
    
    # Train MLP
    print(f"Training MLP Neural Network on {len(X)} samples...")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = mlp_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✓ Training completed!")
    print(f"✓ Test Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(mlp_model, f)
    
    print("✓ Model saved: model.pkl\n")
    
    return {
        'model': mlp_model,
        'word2vec': word2vec_model,
        'accuracy': accuracy,
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

# ==================== MODEL 3: BERT-based Classification ====================

class IntentDataset(Dataset):
    """Custom Dataset for BERT training."""
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_bert_model(original_sentences, labels, classes):
    """
    UNIT 5: BERT-based Intent Classification (Transformers)
    
    BERT (Bidirectional Encoder Representations from Transformers):
    - Pre-trained on large corpus (Wikipedia, BookCorpus)
    - Uses transformer architecture with self-attention
    - Bidirectional: Considers context from both directions
    
    Key Transformer Concepts:
    1. Self-Attention: 
       - Weighs importance of different words in a sentence
       - Q (Query), K (Key), V (Value) matrices
       - Attention(Q,K,V) = softmax(QK^T / √d_k)V
       
    2. Multi-Head Attention:
       - Multiple attention mechanisms in parallel
       - Captures different aspects of relationships
       
    3. Positional Encoding:
       - Adds position information to embeddings
       - Transformers have no inherent sequence order
       
    4. Fine-tuning:
       - Take pre-trained BERT
       - Add classification head
       - Train on task-specific data
       - Transfer learning: leverages learned language understanding
    """
    print("=" * 60)
    print("MODEL 3: BERT-BASED CLASSIFICATION (TRANSFORMER)")
    print("=" * 60)
    
    print("Transformer Architecture Concepts:")
    print("• Self-Attention: Weighs importance of all words for each word")
    print("• Multi-Head Attention: 12 parallel attention mechanisms")
    print("• Positional Encoding: Learns word order relationships")
    print("• Fine-tuning: Adapting pre-trained model to fashion intents\n")
    
    # Create label mapping
    label2id = {label: idx for idx, label in enumerate(classes)}
    id2label = {idx: label for label, idx in label2id.items()}
    numeric_labels = [label2id[label] for label in labels]
    
    # Initialize tokenizer and model
    print("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id
    )
    
    print(f"✓ BERT model loaded with {len(classes)} output classes")
    
    # Split data (handle small datasets without stratification)
    use_stratify = len(numeric_labels) > len(set(numeric_labels)) * 2  # Need at least 2 samples per class
    X_train, X_test, y_train, y_test = train_test_split(
        original_sentences, numeric_labels, test_size=0.2, random_state=42,
        stratify=numeric_labels if use_stratify else None
    )
    
    # Create datasets
    train_dataset = IntentDataset(X_train, y_train, tokenizer)
    test_dataset = IntentDataset(X_test, y_test, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./bert_results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./bert_logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    print("Fine-tuning BERT model...")
    print("This may take several minutes...\n")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating BERT model...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Test Accuracy: {accuracy:.4f}")
    
    # Convert numeric labels back to class names for report
    y_test_labels = [id2label[idx] for idx in y_test]
    y_pred_labels = [id2label[idx] for idx in y_pred]
    
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    # Save model
    model.save_pretrained('./bert_intent_model')
    tokenizer.save_pretrained('./bert_intent_model')
    
    with open('label_mappings.pkl', 'wb') as f:
        pickle.dump({'label2id': label2id, 'id2label': id2label}, f)
    
    print("✓ BERT model saved: ./bert_intent_model\n")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'accuracy': accuracy,
        'report': classification_report(y_test_labels, y_pred_labels, output_dict=True),
        'label2id': label2id,
        'id2label': id2label
    }

# ==================== Model Comparison ====================

def compare_models(tfidf_results, w2v_results, bert_results):
    """Compare all three models."""
    print("=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    
    results = {
        'TF-IDF + Logistic Regression': tfidf_results,
        'Word2Vec + MLP': w2v_results,
        'BERT (Transformer)': bert_results
    }
    
    print(f"{'Model':<30} {'Accuracy':<12} {'Complexity'}")
    print("-" * 60)
    
    for model_name, result in results.items():
        complexity = {
            'TF-IDF + Logistic Regression': 'Low',
            'Word2Vec + MLP': 'Medium',
            'BERT (Transformer)': 'High'
        }[model_name]
        
        print(f"{model_name:<30} {result['accuracy']:<12.4f} {complexity}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    print("• TF-IDF: Fast, interpretable baseline. Good for small datasets.")
    print("• Word2Vec: Captures semantic meaning. Better than bag-of-words.")
    print("• BERT: State-of-the-art. Best for complex language understanding.")
    print("\nRecommendation: Use BERT for production, TF-IDF for quick prototyping.")
    print("=" * 60 + "\n")
    
    # Save comparison
    comparison = {
        model: {'accuracy': results[model]['accuracy']}
        for model in results
    }
    with open('model_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print("✓ Comparison saved: model_comparison.json\n")

# ==================== Main Training Pipeline ====================

def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("FASHION BOT - ADVANCED MODEL TRAINING")
    print("Units 3, 4, 5: Complete NLP Pipeline")
    print("=" * 60 + "\n")
    
    # Load data
    sentences, original_sentences, labels, classes, intents = load_and_prepare_data()
    
    # Save classes
    with open('classes.pkl', 'wb') as f:
        pickle.dump(classes, f)
    
    # Train all models
    print("Training 3 models for comparison...\n")
    
    try:
        # Model 1: TF-IDF + LR
        tfidf_results = train_tfidf_model(original_sentences, labels, classes)
        
        # Model 2: Word2Vec + MLP
        w2v_results = train_word2vec_model(sentences, labels, classes)
        
        # Model 3: BERT
        bert_results = train_bert_model(original_sentences, labels, classes)
        
        # Compare models
        compare_models(tfidf_results, w2v_results, bert_results)
        
        print("=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print("Models saved:")
        print("  ✓ tfidf_model.pkl")
        print("  ✓ model.pkl (Word2Vec + MLP)")
        print("  ✓ bert_intent_model/ (BERT)")
        print("  ✓ classes.pkl")
        print("  ✓ model_comparison.json")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Note: BERT training requires GPU or significant CPU time.")
        print("You can continue with TF-IDF and Word2Vec models.\n")

if __name__ == "__main__":
    main()
