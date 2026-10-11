import json
import numpy as np
import pickle
from gensim.models import Word2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nlp_utils import preprocess_text, get_sentence_vector

def train_models():
    print("Loading intents...")
    with open('intents.json', 'r') as f:
        intents = json.load(f)

    # Prepare training data
    sentences = []
    labels = []
    classes = []
    
    print("Preprocessing data...")
    for intent in intents['intents']:
        tag = intent['tag']
        if tag not in classes:
            classes.append(tag)
            
        for pattern in intent['patterns']:
            tokens = preprocess_text(pattern)
            if tokens: # Avoid empty tokens
                sentences.append(tokens)
                labels.append(tag)
    
    # Train Word2Vec Model
    print("Training Word2Vec model...")
    # min_count=1 because our dataset is small
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_model.save("word2vec.model")
    print("Word2Vec model saved.")

    # Convert sentences to vectors
    X = []
    y = []
    
    for i, tokens in enumerate(sentences):
        # We need to reconstruct the original sentence or just pass tokens if helper allows
        # nlp_utils.get_sentence_vector takes a sentence string and re-processes it.
        # Ideally we should just use the tokens we already have.
        # Let's manually average here for efficiency since we have tokens
        vec = []
        for word in tokens:
            if word in word2vec_model.wv:
                vec.append(word2vec_model.wv[word])
        
        if len(vec) > 0:
            X.append(np.mean(vec, axis=0))
            y.append(labels[i])
        else:
            print(f"Warning: No valid vectors for sentence index {i}")

    X = np.array(X)
    y = np.array(y)

    # Train Classifier
    print(f"Training Classifier on {len(X)} samples...")
    # Using MLPClassifier (Multi-layer Perceptron) for a simple Neural Net approach
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    clf.fit(X, y)
    
    # Save Classifier and Classes
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    with open('classes.pkl', 'wb') as f:
        pickle.dump(classes, f)
        
    print("Model and classes saved.")
    print("Training complete.")

if __name__ == "__main__":
    train_models()
