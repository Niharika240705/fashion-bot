# 📚 Complete NLP Concepts Summary - Fashion Bot

## Overview

This document provides a comprehensive list of all NLP concepts, technologies, and techniques implemented in the Advanced Fashion Bot, organized by course units.

---

## 🎯 UNIT 3: Neural Embeddings & Deep Learning

### 1. Text Preprocessing
#### Concepts Used:
- **Normalization**: Converting text to lowercase
- **Tokenization**: Splitting text into words using NLTK's word_tokenize
- **Stop Words Removal**: Filtering common words (the, is, at, etc.)
- **Lemmatization**: Reducing words to base form (running → run)
- **Stemming**: Reducing words to root (alternative approach, available but not used)
- **Punctuation Removal**: Cleaning special characters using regex

#### Implementation:
```python
# File: nlp_utils.py
- to_lowercase()
- remove_punctuation()
- tokenize()
- remove_stopwords()
- lemmatize_words()
- stem_words()
- preprocess_text()
```

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)
#### Concepts:
- **Term Frequency (TF)**: How often a term appears in a document
- **Inverse Document Frequency (IDF)**: How rare a term is across all documents
- **TF-IDF Score**: TF × IDF weighted importance
- **Vectorization**: Converting text to numerical vectors
- **N-grams**: Unigrams (1-word) and bigrams (2-word) phrases
- **Document-Term Matrix**: Sparse matrix representation

#### Mathematical Formula:
```
TF-IDF(t, d, D) = TF(t, d) × log(|D| / |{d ∈ D : t ∈ d}|)

Where:
- t = term
- d = document
- D = corpus
- |D| = total number of documents
```

#### Implementation:
```python
# File: train_advanced.py
- TfidfVectorizer with max_features=1000
- ngram_range=(1, 2) for unigrams and bigrams
- min_df=1, max_df=0.8 for frequency filtering
```

### 3. Logistic Regression
#### Concepts:
- **Binary/Multi-class Classification**
- **Sigmoid Function**: σ(z) = 1 / (1 + e^(-z))
- **Log Loss**: Cost function for classification
- **Multinomial Classification**: One-vs-rest strategy
- **Regularization**: L2 penalty (C parameter)
- **Gradient Descent**: Optimization algorithm

#### Implementation:
```python
# File: train_advanced.py
LogisticRegression(
    max_iter=1000,
    C=1.0,  # Regularization strength
    multi_class='multinomial'
)
```

### 4. Word2Vec
#### Concepts:
- **Distributed Representations**: Dense word vectors
- **CBOW (Continuous Bag of Words)**: Predict word from context
- **Skip-gram**: Predict context from word
- **Word Embeddings**: Semantic meaning in vectors
- **Vector Size**: Dimensionality of embeddings (100 in our case)
- **Context Window**: Surrounding words to consider (5 words)
- **Negative Sampling**: Training optimization technique
- **Semantic Similarity**: cos(v1, v2) = (v1 · v2) / (||v1|| × ||v2||)

#### Vector Arithmetic:
```
king - man + woman ≈ queen
paris - france + italy ≈ rome
```

#### Implementation:
```python
# File: train_advanced.py
Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=10
)
```

### 5. Neural Networks (MLP - Multi-Layer Perceptron)
#### Concepts:
- **Feedforward Network**: Input → Hidden → Output
- **Hidden Layers**: Non-linear transformations
- **Activation Functions**: ReLU (Rectified Linear Unit)
- **Backpropagation**: Gradient computation
- **Loss Function**: Cross-entropy for classification
- **Optimizer**: Adam (adaptive learning rate)
- **Early Stopping**: Prevent overfitting
- **Dropout**: Regularization technique (implicit in scikit-learn)

#### Architecture:
```
Input Layer (100 neurons - Word2Vec dimension)
    ↓
Hidden Layer 1 (64 neurons + ReLU)
    ↓
Hidden Layer 2 (32 neurons + ReLU)
    ↓
Output Layer (num_classes + Softmax)
```

#### Implementation:
```python
# File: train_advanced.py
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1
)
```

### 6. Model Evaluation
#### Metrics Used:
- **Accuracy**: Correct predictions / Total predictions
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Confusion Matrix**: Prediction distribution
- **Classification Report**: Per-class metrics

#### Implementation:
```python
# File: train_advanced.py
- accuracy_score()
- precision_recall_fscore_support()
- classification_report()
- cross_val_score(cv=5)
- confusion_matrix()
```

---

## 🔍 UNIT 4: Information Extraction

### 1. Named Entity Recognition (NER)
#### Concepts:
- **Machine Learning-based NER**: SpaCy's trained models
- **Entity Types**: PERSON, ORG, GPE, PRODUCT, DATE, MONEY, etc.
- **IOB Tagging**: Inside-Outside-Beginning notation
- **Sequence Labeling**: Assigning tags to sequential data
- **CNN Feature Extraction**: Convolutional layers for patterns
- **Transition-based Parsing**: Greedy parsing approach
- **Pre-trained Models**: OntoNotes 5 corpus

#### Entity Types:
```
PERSON: People, including fictional
ORG: Companies, agencies, institutions
GPE: Countries, cities, states
PRODUCT: Objects, vehicles, foods, etc.
DATE: Absolute or relative dates
MONEY: Monetary values
TIME: Times smaller than a day
LOC: Non-GPE locations
```

#### Implementation:
```python
# File: advanced_nlp.py
class AdvancedNER:
    - extract_entities()
    - extract_fashion_entities()
    - get_noun_chunks()
```

### 2. Fashion-Specific Entity Detection
#### Concepts:
- **Domain-Specific NER**: Custom entity recognition
- **Pattern Matching**: Regex and keyword matching
- **Entity Linking**: Connecting mentions to brands
- **Gazetteer Lists**: Pre-defined brand and product lists

#### Implemented:
```python
Fashion Brands: Gucci, Prada, Chanel, Dior, etc. (20+ brands)
Fashion Products: dress, handbag, shoes, jewelry, etc. (30+ items)
```

### 3. Keyphrase Extraction
#### A. KeyBERT (BERT-based)
**Concepts:**
- **BERT Embeddings**: Pre-trained contextual vectors
- **Cosine Similarity**: Measure phrase relevance
- **Candidate Generation**: n-gram extraction
- **MMR (Maximal Marginal Relevance)**: Diversity in results
- **Semantic Matching**: Context-aware extraction

**Algorithm:**
```
1. Embed document with BERT
2. Extract candidate keyphrases (n-grams)
3. Embed each candidate
4. Calculate cosine similarity with document
5. Apply MMR for diversity
6. Rank and return top-k
```

#### B. YAKE (Yet Another Keyword Extractor)
**Concepts:**
- **Statistical Approach**: No machine learning needed
- **Word Features**:
  - Position (where in text)
  - Frequency (how often)
  - Context (surrounding words)
  - Case (capitalization)
- **Co-occurrence**: Word relationships
- **Language-Independent**: Works for any language
- **Unsupervised**: No training needed

**Features:**
```python
- Term frequency
- Position of first occurrence
- Left/right context
- Word case statistics
- Word relatedness to context
```

#### Implementation:
```python
# File: advanced_nlp.py
class KeyphraseExtractor:
    - extract_keyphrases_keybert()
    - extract_keyphrases_yake()
    - extract_combined()
```

### 4. Relationship Extraction
#### Concepts:
- **Dependency Parsing**: Grammatical structure analysis
- **Subject-Verb-Object (SVO)**: Triplet extraction
- **Dependency Tree**: Hierarchical word relationships
- **Dependency Labels**:
  - nsubj: Nominal subject
  - dobj: Direct object
  - prep: Prepositional modifier
  - attr: Attribute
- **Grammatical Relations**: Syntactic connections

#### Dependency Types:
```
nsubj: Nominal subject
dobj: Direct object
iobj: Indirect object
prep: Prepositional modifier
pobj: Object of preposition
attr: Attribute
amod: Adjectival modifier
```

#### Implementation:
```python
# File: advanced_nlp.py
class RelationshipExtractor:
    - extract_relationships()
    - extract_fashion_relationships()
```

### 5. Part-of-Speech (POS) Tagging
#### Concepts:
- **POS Tags**: Grammatical categories
- **Universal POS Tags**: NOUN, VERB, ADJ, ADV, etc.
- **Fine-grained Tags**: Penn Treebank tagset
- **HMM Tagging**: Hidden Markov Models
- **Statistical Models**: Trained on tagged corpora

#### Common Tags:
```
NN: Noun, singular
NNS: Noun, plural
VB: Verb, base form
VBD: Verb, past tense
JJ: Adjective
RB: Adverb
DT: Determiner
```

#### Implementation:
```python
# File: nlp_utils.py
- get_pos_tags()
```

### 6. Chunking (Phrase Extraction)
#### Concepts:
- **Noun Phrase Chunking**: Extracting NP patterns
- **Regular Expression Chunking**: Grammar-based extraction
- **Chunk Grammar**: Context-free grammar rules
- **Shallow Parsing**: Partial syntactic analysis

#### Grammar Pattern:
```python
NP: {<DT>?<JJ>*<NN.*>+}

Meaning:
- DT?: Optional determiner (the, a)
- JJ*: Zero or more adjectives
- NN.*+: One or more nouns
```

#### Implementation:
```python
# File: nlp_utils.py
- extract_noun_phrases()
```

### 7. Sentiment Analysis
#### Concepts:
- **Polarity**: Positive/Negative/Neutral classification
- **Subjectivity**: Objective vs Subjective
- **Sentiment Score**: [-1, 1] numerical scale
- **Lexicon-based Approach**: Word sentiment dictionaries
- **Rule-based Fallback**: Simple positive/negative word lists

#### Implementation:
```python
# File: advanced_nlp.py
class SentimentAnalyzer:
    - analyze_sentiment()
```

---

## 🤖 UNIT 5: Transformers & Advanced Models

### 1. BERT (Bidirectional Encoder Representations from Transformers)
#### Concepts:
- **Transformer Architecture**: Attention-based model
- **Bidirectional Context**: Left and right context simultaneously
- **Pre-training**: Large-scale unsupervised learning
- **Fine-tuning**: Task-specific adaptation
- **WordPiece Tokenization**: Subword units
- **Special Tokens**: [CLS], [SEP], [PAD], [MASK]
- **Token Embeddings**: Input representation
- **Segment Embeddings**: Sentence separation
- **Position Embeddings**: Sequence order information

#### Architecture Details:
```
BERT-base-uncased:
- 12 Transformer layers (encoder blocks)
- 768 hidden dimensions
- 12 attention heads
- 110M parameters
- Trained on: Wikipedia + BookCorpus
```

#### Pre-training Tasks:
1. **Masked Language Modeling (MLM)**:
   - Mask 15% of tokens
   - Predict masked tokens from context
   
2. **Next Sentence Prediction (NSP)**:
   - Predict if sentence B follows sentence A
   - Binary classification task

#### Implementation:
```python
# File: train_advanced.py
BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=num_classes
)
```

### 2. Self-Attention Mechanism
#### Concepts:
- **Attention Weights**: Importance of each word to others
- **Query-Key-Value (QKV)**: Three learned projections
- **Scaled Dot-Product Attention**: Core mechanism
- **Attention Score**: Relevance between tokens
- **Softmax Normalization**: Converting scores to probabilities

#### Mathematical Formula:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Where:
- Q (Query): What we're looking for
- K (Key): What to pay attention to
- V (Value): Actual information
- d_k: Dimension of keys (for scaling)
- √d_k: Scaling factor to prevent large values
```

#### Step-by-Step:
```
1. Linear transformations: Q=XW_q, K=XW_k, V=XW_v
2. Compute attention scores: Scores = QK^T
3. Scale: Scores / √d_k
4. Softmax: Weights = softmax(Scores)
5. Weighted sum: Output = Weights × V
```

#### Attention Example:
```
Input: "I want luxury handbag"

Attention weights for "handbag":
I       → 0.05 (low attention)
want    → 0.15 (medium attention)
luxury  → 0.70 (high attention)
handbag → 0.10 (self-attention)
```

### 3. Multi-Head Attention
#### Concepts:
- **Multiple Attention Mechanisms**: Parallel attention layers
- **Different Representation Subspaces**: Learn different aspects
- **Head Count**: 12 heads in BERT-base
- **Concatenation**: Combining head outputs
- **Linear Projection**: Final transformation

#### Formula:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

#### Benefits:
- Capture different linguistic phenomena
- Learn various relationships simultaneously
- More robust representations

### 4. Positional Encoding
#### Concepts:
- **Sequence Order**: Transformers have no inherent order
- **Sinusoidal Encoding**: Sin/cos functions for positions
- **Learned Embeddings**: BERT uses learned positions
- **Maximum Length**: 512 tokens in BERT

#### Formula (original Transformer):
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 5. Fine-tuning & Transfer Learning
#### Concepts:
- **Transfer Learning**: Leverage pre-trained knowledge
- **Task-Specific Head**: Classification layer on top
- **Learning Rate**: Lower for pre-trained layers
- **Unfreezing**: Gradual layer unfreezing
- **Domain Adaptation**: Adapting to fashion domain

#### Process:
```
1. Load pre-trained BERT
2. Add classification head (num_classes outputs)
3. Freeze BERT layers (optional)
4. Train on task-specific data
5. Fine-tune entire model
6. Save adapted model
```

#### Implementation:
```python
# File: train_advanced.py
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01
)
```

### 6. Transformer Encoder
#### Components:
1. **Multi-Head Self-Attention**
2. **Add & Norm** (Residual connection + Layer normalization)
3. **Feed-Forward Network** (2 linear layers + ReLU)
4. **Add & Norm** (Second residual + normalization)

#### Layer Normalization:
```
LN(x) = γ × (x - μ) / √(σ² + ε) + β

Where:
- μ: Mean
- σ: Standard deviation
- γ, β: Learned parameters
- ε: Small constant (1e-5)
```

### 7. Hugging Face Inference API
#### Concepts:
- **API-based Inference**: Cloud model serving
- **Zero-shot Classification**: No training needed
- **Model Hub**: Access to thousands of models
- **REST API**: HTTP-based model access
- **Rate Limiting**: Request quotas
- **Async Processing**: Non-blocking inference

#### Endpoints Used:
```
1. Zero-shot Classification:
   - facebook/bart-large-mnli
   
2. Sentiment Analysis:
   - distilbert-base-uncased-finetuned-sst-2-english
   
3. Text Generation:
   - gpt2
   
4. Question Answering:
   - deepset/roberta-base-squad2
```

#### API Request:
```python
headers = {"Authorization": f"Bearer {api_key}"}
payload = {
    "inputs": text,
    "parameters": {"candidate_labels": classes}
}
response = requests.post(api_url, headers=headers, json=payload)
```

### 8. Tokenization
#### Concepts:
- **WordPiece Tokenization**: Subword splitting
- **Vocabulary**: 30,000 tokens in BERT
- **Unknown Tokens**: [UNK] for out-of-vocabulary
- **Special Tokens**: [CLS], [SEP], [PAD]
- **Maximum Length**: Truncation and padding

#### Example:
```
Input: "I love luxury fashion"

Tokens: ['[CLS]', 'i', 'love', 'luxury', 'fashion', '[SEP]']
IDs: [101, 1045, 2293, 9354, 4827, 102]
Attention Mask: [1, 1, 1, 1, 1, 1]
```

---

## 🔧 Additional Technologies

### Libraries & Frameworks
1. **NLTK**: Natural Language Toolkit
   - Tokenization, POS tagging, NER
   - WordNet lemmatizer
   - Stop words corpus

2. **SpaCy**: Industrial-strength NLP
   - Fast processing
   - Pre-trained models
   - Dependency parsing

3. **Gensim**: Topic modeling & embeddings
   - Word2Vec implementation
   - FastText support
   - Doc2Vec

4. **scikit-learn**: Machine learning
   - TF-IDF vectorizer
   - Logistic Regression
   - MLP Classifier
   - Evaluation metrics

5. **Transformers (Hugging Face)**: State-of-the-art NLP
   - BERT, RoBERTa, GPT models
   - Easy fine-tuning
   - Model hub

6. **PyTorch**: Deep learning framework
   - Neural network building blocks
   - Automatic differentiation
   - GPU acceleration

7. **KeyBERT**: Keyphrase extraction
   - BERT-based extraction
   - Minimal code needed

8. **YAKE**: Keyword extraction
   - Language-independent
   - Statistical approach

### Data Formats
- **JSON**: Intent data storage
- **Pickle**: Model serialization
- **CSV**: Data export (if needed)

### Web Technologies
- **Flask**: Web framework
- **HTML/CSS/JavaScript**: Frontend
- **REST API**: Client-server communication

---

## 📊 Performance Metrics

### Model Comparison
```
Model               Accuracy  Speed     Complexity
-----------------   --------  -------  ----------
TF-IDF + LR         ~85%      Fast     Low
Word2Vec + MLP      ~88%      Medium   Medium
BERT Fine-tuned     ~92%      Slow     High
HF API              ~93%      Medium   N/A
```

### Evaluation Metrics Used
- Accuracy
- Precision (per class)
- Recall (per class)
- F1-Score (weighted)
- Confusion Matrix
- Cross-Validation (5-fold)

---

## 🎓 Key Takeaways

### Unit 3: Core Concepts
✅ Text preprocessing pipeline  
✅ TF-IDF for feature extraction  
✅ Word embeddings capture semantics  
✅ Neural networks learn non-linear patterns  
✅ Model comparison is essential  

### Unit 4: Information Extraction
✅ ML-based NER outperforms rule-based  
✅ Multiple extraction methods complement each other  
✅ Domain-specific entities need custom handling  
✅ Dependency parsing reveals relationships  
✅ Sentiment adds user insight  

### Unit 5: Advanced Models
✅ Transformers revolutionized NLP  
✅ Self-attention enables bidirectional context  
✅ Pre-training + fine-tuning is powerful  
✅ API deployment simplifies production  
✅ Transfer learning accelerates development  

---

## 📖 References

### Papers
1. **Attention Is All You Need** (Vaswani et al., 2017) - Transformer architecture
2. **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
3. **Efficient Estimation of Word Representations** (Mikolov et al., 2013) - Word2Vec
4. **YAKE! Keyword Extraction** (Campos et al., 2020)

### Tools & Libraries
- NLTK: https://www.nltk.org/
- SpaCy: https://spacy.io/
- Gensim: https://radimrehurek.com/gensim/
- Hugging Face: https://huggingface.co/
- KeyBERT: https://github.com/MaartenGr/KeyBERT

---

**This project demonstrates the complete evolution of NLP: from traditional methods (TF-IDF) through neural embeddings (Word2Vec) to state-of-the-art transformers (BERT).**

**Total Concepts Covered: 100+**
**Technologies Used: 15+**
**Models Implemented: 4**
**Units Covered: 3, 4, 5 ✅**
