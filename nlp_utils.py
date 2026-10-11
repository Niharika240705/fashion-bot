import nltk
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def to_lowercase(text):
    """Converts text to lowercase."""
    return text.lower()

def remove_punctuation(text):
    """Removes non-alphanumeric characters."""
    return re.sub(r'[^\w\s]', '', text)

def tokenize(text):
    """Tokenizes text into words."""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Removes stop words from a list of tokens."""
    return [word for word in tokens if word not in stop_words]

def stem_words(tokens):
    """Stems a list of tokens."""
    return [stemmer.stem(word) for word in tokens]

def lemmatize_words(tokens):
    """Lemmatizes a list of tokens."""
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(text):
    """
    Full preprocessing pipeline:
    Lower -> Remove Punctuation -> Tokenize -> Remove Stopwords -> Lemmatize
    (We choose lemmatization over stemming for better semantic understanding in this context, 
    but stemming is available if needed)
    """
    text = to_lowercase(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    # Using lemmatization as it generally produces more readable root words for embeddings
    tokens = lemmatize_words(tokens) 
    return tokens

def get_sentence_vector(sentence, model):
    """
    Generates a vector for a sentence by averaging the word vectors of its tokens.
    'model' is a trained Word2Vec model.
    """
    tokens = preprocess_text(sentence)
    word_vectors = []
    for token in tokens:
        if token in model.wv:
            word_vectors.append(model.wv[token])
    
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    
    return np.mean(word_vectors, axis=0)

def get_pos_tags(text):
    """Returns Part-of-Speech tags for the input text."""
    tokens = tokenize(text)
    return nltk.pos_tag(tokens)

def get_named_entities(text):
    """Extracts named entities using NLTK's ne_chunk."""
    pos_tags = get_pos_tags(text)
    tree = nltk.ne_chunk(pos_tags)
    entities = []
    for subtree in tree:
        if hasattr(subtree, 'label'):
            entity = " ".join([leaf[0] for leaf in subtree])
            entities.append((entity, subtree.label()))
    return entities

def extract_noun_phrases(text):
    """Extracts potential product noun phrases using chunking."""
    tokens = tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    # Grammar: Optional Determiner + Optional Adjectives + Noun(s)
    grammar = "NP: {<DT>?<JJ>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(pos_tags)
    
    phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            phrase = " ".join([leaf[0] for leaf in subtree])
            phrases.append(phrase)
    return phrases
