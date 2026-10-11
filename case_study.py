import json
import numpy as np
import time
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import nltk
from nltk import pos_tag, ne_chunk
from nlp_utils import preprocess_text, to_lowercase, remove_punctuation, tokenize, remove_stopwords, stem_words, lemmatize_words

# Ensure NLTK data is downloaded (redundant but safe)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('maxent_ne_chunker_tab')

def load_data():
    with open('intents.json', 'r') as f:
        data = json.load(f)
    samples = []
    labels = []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            samples.append(pattern)
            labels.append(intent['tag'])
    return samples, labels

def task_1_preprocessing(samples):
    print("\n" + "="*50)
    print("TASK 1: Text Pre-processing")
    print("="*50)
    
    # Select 5 diverse example sentences
    indices = [0, 5, 10, 15, 20] # Adjust if dataset is smaller
    examples = [samples[i] for i in indices if i < len(samples)]
    
    print(f"{'Original':<50} | {'Processed (Steps Breakdown)':<50}")
    print("-" * 105)
    
    for text in examples:
        # Step-by-step
        lower = to_lowercase(text)
        no_punct = remove_punctuation(lower)
        tokens = tokenize(no_punct)
        no_stop = remove_stopwords(tokens)
        lemmatized = lemmatize_words(no_stop)
        stemmed = stem_words(no_stop)
        
        print(f"Original: {text}")
        print(f"1. Lowercase: {lower}")
        print(f"2. No Punctuation: {no_punct}")
        print(f"3. Tokenized: {tokens}")
        print(f"4. No Stopwords: {no_stop}")
        print(f"5. Lemmatized: {lemmatized}")
        print(f"   (Stemmed): {stemmed}")
        print("-" * 50)

def task_2_feature_extraction(samples):
    print("\n" + "="*50)
    print("TASK 2: Feature Extraction Techniques")
    print("="*50)

    # 1. Bag of Words
    print("\n--- 2.1 Bag of Words (BoW) ---")
    vectorizer_bow = CountVectorizer()
    X_bow = vectorizer_bow.fit_transform(samples)
    print(f"BoW Vocabulary Size: {len(vectorizer_bow.get_feature_names_out())}")
    print(f"Example Feature Names: {vectorizer_bow.get_feature_names_out()[:10]}")
    print("Explanation: BoW represents text as the frequency of words, ignoring order.")

    # 2. TF-IDF
    print("\n--- 2.2 TF-IDF ---")
    vectorizer_tfidf = TfidfVectorizer()
    X_tfidf = vectorizer_tfidf.fit_transform(samples)
    print(f"TF-IDF Shape: {X_tfidf.shape}")
    print("Explanation: TF-IDF weighs terms by importance (rare words get higher weights).")

    # 3. N-grams
    print("\n--- 2.3 N-grams (Bigrams) ---")
    vectorizer_ngram = CountVectorizer(ngram_range=(2, 2))
    X_ngram = vectorizer_ngram.fit_transform(samples)
    print(f"Bigram Vocabulary Size: {len(vectorizer_ngram.get_feature_names_out())}")
    print(f"Example Bigrams: {vectorizer_ngram.get_feature_names_out()[:10]}")
    print("Explanation: N-grams capture local context by grouping adjacent words.")

    # 4. Word Embeddings (Word2Vec)
    print("\n--- 2.4 Word Embeddings (Word2Vec) ---")
    # Preprocess all samples for Word2Vec
    tokenized_samples = [preprocess_text(s) for s in samples]
    # Train a small Word2Vec model
    w2v_model = Word2Vec(sentences=tokenized_samples, vector_size=100, window=5, min_count=1, workers=4)
    print(f"Word2Vec Model Trained. Vector size: {w2v_model.vector_size}")
    if 'gucci' in w2v_model.wv:
        print(f"Vector for 'gucci' (first 5 dims): {w2v_model.wv['gucci'][:5]}")
    print("Explanation: Embeddings capture semantic meaning and relationships between words.")
    
    return X_bow, X_tfidf, X_ngram, w2v_model, tokenized_samples

def task_3_feature_engineering(samples):
    print("\n" + "="*50)
    print("TASK 3: Feature Engineering")
    print("="*50)
    
    # Custom features:
    # 1. Length of sentence (word count)
    # 2. Presence of luxury brand names (binary)
    # 3. Presence of question mark (binary check on original text)
    
    luxury_brands = ['gucci', 'prada', 'chanel', 'dior', 'hermes', 'lv', 'louis', 'vuitton', 'role', 'cartier']
    
    custom_features = []
    for text in samples:
        features = []
        lower_text = text.lower()
        
        # Feature 1: Word Count
        features.append(len(text.split()))
        
        # Feature 2: Brand Presence
        has_brand = 1 if any(brand in lower_text for brand in luxury_brands) else 0
        features.append(has_brand)
        
        # Feature 3: Is Question
        is_question = 1 if '?' in text else 0
        features.append(is_question)
        
        custom_features.append(features)
    
    print("Created 3 Custom Features:")
    print("1. Sentence Length (Word Count)")
    print("2. Luxury Brand Presence (Binary)")
    print("3. Is Question (Binary)")
    print(f"Example features for '{samples[0]}': {custom_features[0]}")
    
    return np.array(custom_features)

def train_evaluate(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    
    return acc, train_time

def get_sentence_embeddings(tokenized_samples, w2v_model):
    X_w2v = []
    for tokens in tokenized_samples:
        vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if vecs:
            X_w2v.append(np.mean(vecs, axis=0))
        else:
            X_w2v.append(np.zeros(w2v_model.vector_size))
    return np.array(X_w2v)

def task_4_comparison(samples, labels, X_bow, X_tfidf, X_ngram, w2v_model, tokenized_samples, X_custom):
    print("\n" + "="*50)
    print("TASK 4: Feature Comparison")
    print("="*50)
    
    X_w2v = get_sentence_embeddings(tokenized_samples, w2v_model)
    
    # Evaluation
    results = {}
    results['BoW'] = train_evaluate(X_bow, labels, 'BoW')
    results['TF-IDF'] = train_evaluate(X_tfidf, labels, 'TF-IDF')
    results['N-grams'] = train_evaluate(X_ngram, labels, 'N-grams')
    results['Word2Vec'] = train_evaluate(X_w2v, labels, 'Word2Vec')
    results['Custom'] = train_evaluate(X_custom, labels, 'Custom') # For interest
    
    print(f"{'Feature Set':<15} | {'Accuracy':<10} | {'Time (s)':<10}")
    print("-" * 45)
    for name, (acc, t) in results.items():
        print(f"{name:<15} | {acc:.4f}     | {t:.4f}")

    print("\nComparison Analysis:")
    print("- Context Capture: N-grams capture local context. Word2Vec captures semantic context.")
    print("- Synonym Handling: Word2Vec handles synonyms best (similar vectors). BoW/TF-IDF treat them as distinct.")
    print("- Memory: BoW/TF-IDF are sparse (efficient for large vocab). Word2Vec is dense (fixed size).")
    print("- Performance: Embeddings often outperform on semantic tasks, but BoW is strong on keyword matching.")

def task_5_model_analysis():
    print("\n" + "="*50)
    print("TASK 5: Model Impact Analysis")
    print("="*50)
    print("Recommendation: For this specific chatbot, valid TF-IDF or Word2Vec are likely best.")
    print("Analysis: Word embeddings (Word2Vec) generally outperform TF-IDF in semantic tasks because they capture the MEANING of words, not just their presence.")
    print("Example: 'jacket' and 'coat' are close in vector space, but orthogonal in TF-IDF. This allows the bot to understand 'coat' even if it only saw 'jacket' in training (if pretrained appropriately).")

def task_6_real_world():
    print("\n" + "="*50)
    print("TASK 6: Real-World Extension")
    print("="*50)
    print("1. Multilingual Support: Use multilingual embeddings (e.g., mBERT, XLM-R) or translate inputs to English before processing.")
    print("2. Spelling Mistakes: Implement a spell checker (pyspellchecker) or use sub-word embeddings (FastText) which are robust to typos.")
    print("3. Slang Handling: Fine-tune embeddings on social media datasets (Twitter/Reddit) or maintain a custom slang dictionary/mapping.")

def task_7_pos_ner(samples):
    print("\n" + "="*50)
    print("TASK 7: POS Tagging & NER")
    print("="*50)
    
    # Sample sentences for demonstration
    demo_sentences = [
        "I want to buy a Gucci bag in Paris.",
        "Does Christian Louboutin make sneakers?",
        "Styling advice for a summer wedding in Italy."
    ]
    
    print(f"{'Sentence':<50}")
    print("-" * 50)
    
    for sentence in demo_sentences:
        print(f"\nProcessing: '{sentence}'")
        tokens = nltk.word_tokenize(sentence)
        
        # POS Tagging
        pos_tags = nltk.pos_tag(tokens)
        print("POS Tags:", pos_tags)
        
        # Named Entity Recognition (NER)
        ner_tree = nltk.ne_chunk(pos_tags)
        print("NER Tree:", ner_tree)
        
        # Extract entities
        entities = []
        for subtree in ner_tree:
            if hasattr(subtree, 'label'):
                entity_name = " ".join([leaf[0] for leaf in subtree])
                entities.append((entity_name, subtree.label()))
        
        if entities:
            print("Detected Entities:", entities)
        else:
            print("No Named Entities detected.")

def task_8_information_retrieval(samples, labels):
    print("\n" + "="*50)
    print("TASK 8: Information Retrieval (IR)")
    print("="*50)
    
    # Create TF-IDF Corpus
    vectorizer = TfidfVectorizer()
    data_vectors = vectorizer.fit_transform(samples)
    
    # Demo Queries
    queries = [
        "show me luxury shoes",
        "who is Prada?",
        "investment bags"
    ]
    
    for query in queries:
        query_vec = vectorizer.transform([query])
        
        # Calculate Cosine Similarity
        cosine_similarities = cosine_similarity(query_vec, data_vectors).flatten()
        
        # Get Top Match
        best_idx = cosine_similarities.argmax()
        best_score = cosine_similarities[best_idx]
        best_match_text = samples[best_idx]
        best_match_intent = labels[best_idx]
        
        print(f"\nQuery: '{query}'")
        print(f"Best Match (Score: {best_score:.4f}): '{best_match_text}' (Intent: {best_match_intent})")

def task_9_information_extraction(samples):
    print("\n" + "="*50)
    print("TASK 9: Information Extraction (IE)")
    print("="*50)
    
    # Using chunking to extract Noun Phrases (NP) that might represent products
    # Grammar: Optional Determiner (DT) + Any number of Adjectives (JJ) + Noun (NN)
    grammar = "NP: {<DT>?<JJ>*<NN.*>+}"
    chunk_parser = nltk.RegexpParser(grammar)
    
    demo_sentences = [
        "Show me a red leather bag.",
        "I am looking for stylish summer dresses.",
        "Buy a vintage Chanel scarf."
    ]
    
    for sentence in demo_sentences:
        print(f"\nProcessing: '{sentence}'")
        tokens = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)
        tree = chunk_parser.parse(pos_tags)
        
        extracted_products = []
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                product = " ".join([leaf[0] for leaf in subtree])
                extracted_products.append(product)
                
        print(f"Extracted Noun Phrases (Potential Products): {extracted_products}")

def main():
    samples, labels = load_data()
    
    task_1_preprocessing(samples)
    X_bow, X_tfidf, X_ngram, w2v_model, tokenized_samples = task_2_feature_extraction(samples)
    X_custom = task_3_feature_engineering(samples)
    task_4_comparison(samples, labels, X_bow, X_tfidf, X_ngram, w2v_model, tokenized_samples, X_custom)
    task_5_model_analysis()
    task_6_real_world()
    
    # New Advanced NLP Tasks
    task_7_pos_ner(samples)
    task_8_information_retrieval(samples, labels)
    task_9_information_extraction(samples)

if __name__ == "__main__":
    main()
