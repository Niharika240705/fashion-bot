"""
Advanced NLP Utilities for Fashion Bot (Unit 4)
Implements:
1. SpaCy-based Named Entity Recognition (improved NER)
2. Keyphrase Extraction (KeyBERT, YAKE)
3. Relationship Extraction
4. Fashion-specific entity detection
"""

import spacy
from keybert import KeyBERT
import yake
import re
from collections import defaultdict

# ==================== SpaCy NER (Unit 4) ====================

class AdvancedNER:
    """
    UNIT 4: Advanced Named Entity Recognition using SpaCy
    
    Advantages over NLTK NER:
    - Uses machine learning (CNN + entity linking)
    - Better accuracy with pre-trained models
    - Recognizes more entity types
    - Industry-standard for production NER
    
    SpaCy NER uses:
    - Transition-based parsing
    - CNN for feature extraction
    - BiLSTM for sequence tagging
    - Pre-trained on OntoNotes 5 corpus
    """
    
    def __init__(self, model_name='en_core_web_sm'):
        """Initialize SpaCy NER model."""
        try:
            self.nlp = spacy.load(model_name)
            print(f"✓ SpaCy model '{model_name}' loaded successfully")
        except OSError:
            print(f"Downloading SpaCy model '{model_name}'...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', model_name])
            self.nlp = spacy.load(model_name)
            print(f"✓ SpaCy model '{model_name}' loaded successfully")
        
        # Fashion-specific brand patterns
        self.fashion_brands = [
            'gucci', 'prada', 'chanel', 'dior', 'versace', 'armani',
            'louis vuitton', 'hermes', 'burberry', 'fendi', 'valentino',
            'givenchy', 'balenciaga', 'dolce gabbana', 'yves saint laurent',
            'bottega veneta', 'celine', 'cartier', 'bulgari', 'tiffany'
        ]
        
        # Fashion product categories
        self.fashion_products = [
            'dress', 'gown', 'suit', 'blazer', 'jacket', 'coat', 'pants',
            'trousers', 'jeans', 'shirt', 'blouse', 'sweater', 'cardigan',
            'skirt', 'handbag', 'purse', 'wallet', 'shoes', 'heels',
            'boots', 'sneakers', 'sandals', 'watch', 'jewelry', 'necklace',
            'bracelet', 'earrings', 'ring', 'sunglasses', 'belt', 'scarf',
            'tie', 'hat', 'gloves'
        ]
    
    def extract_entities(self, text):
        """
        Extract named entities using SpaCy's ML-based NER.
        
        Returns entities with types:
        PERSON, ORG, GPE, PRODUCT, MONEY, DATE, etc.
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def extract_fashion_entities(self, text):
        """
        Extract fashion-specific entities (brands, products).
        Combines SpaCy NER with custom pattern matching.
        """
        doc = self.nlp(text.lower())
        entities = {'brands': [], 'products': [], 'general': []}
        
        # Get SpaCy entities
        for ent in doc.ents:
            entities['general'].append({
                'text': ent.text,
                'type': ent.label_
            })
        
        # Extract fashion brands
        text_lower = text.lower()
        for brand in self.fashion_brands:
            if brand in text_lower:
                entities['brands'].append(brand.title())
        
        # Extract fashion products
        for token in doc:
            if token.text in self.fashion_products:
                entities['products'].append(token.text)
        
        # Remove duplicates
        entities['brands'] = list(set(entities['brands']))
        entities['products'] = list(set(entities['products']))
        
        return entities
    
    def get_noun_chunks(self, text):
        """Extract meaningful noun phrases."""
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]

# ==================== Keyphrase Extraction (Unit 4) ====================

class KeyphraseExtractor:
    """
    UNIT 4: Keyphrase Extraction using KeyBERT and YAKE
    
    KeyBERT:
    - Uses BERT embeddings to find keyphrases
    - Calculates cosine similarity between document and candidate phrases
    - Context-aware, semantic-based extraction
    
    YAKE (Yet Another Keyword Extractor):
    - Statistical approach
    - Uses word features: position, frequency, co-occurrence
    - Language-independent, unsupervised
    - Good for quick extraction without ML models
    """
    
    def __init__(self):
        """Initialize keyphrase extractors."""
        self.keybert = KeyBERT()
        self.yake_extractor = yake.KeywordExtractor(
            lan="en",
            n=2,  # Extract up to bigrams
            dedupLim=0.9,
            top=10
        )
        print("✓ Keyphrase extractors initialized")
    
    def extract_keyphrases_keybert(self, text, top_n=5):
        """
        Extract keyphrases using KeyBERT (BERT-based).
        Returns phrases with relevance scores.
        """
        try:
            keyphrases = self.keybert.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n,
                use_mmr=True,  # Maximal Marginal Relevance for diversity
                diversity=0.5
            )
            return [{'phrase': kp[0], 'score': kp[1]} for kp in keyphrases]
        except:
            return []
    
    def extract_keyphrases_yake(self, text):
        """
        Extract keyphrases using YAKE (statistical).
        Returns phrases with scores (lower is better).
        """
        keywords = self.yake_extractor.extract_keywords(text)
        return [{'phrase': kp[1], 'score': kp[0]} for kp in keywords]
    
    def extract_combined(self, text, top_n=5):
        """Combine both methods for robust extraction."""
        keybert_phrases = self.extract_keyphrases_keybert(text, top_n)
        yake_phrases = self.extract_keyphrases_yake(text)[:top_n]
        
        return {
            'keybert': keybert_phrases,
            'yake': yake_phrases
        }

# ==================== Relationship Extraction (Unit 4) ====================

class RelationshipExtractor:
    """
    UNIT 4: Simple Relationship Extraction
    
    Extracts relationships between entities in text.
    Uses dependency parsing to find subject-verb-object patterns.
    
    Dependency Parsing:
    - Analyzes grammatical structure
    - Identifies relationships between words
    - Uses SpaCy's trained parser
    """
    
    def __init__(self):
        """Initialize SpaCy for dependency parsing."""
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
        
        print("✓ Relationship extractor initialized")
    
    def extract_relationships(self, text):
        """
        Extract subject-verb-object relationships.
        Returns triplets: (subject, relation, object)
        """
        doc = self.nlp(text)
        relationships = []
        
        for token in doc:
            # Look for verbs
            if token.pos_ == 'VERB':
                subject = None
                obj = None
                
                # Find subject
                for child in token.children:
                    if child.dep_ in ['nsubj', 'nsubjpass']:
                        subject = child.text
                    elif child.dep_ in ['dobj', 'attr', 'prep']:
                        obj = child.text
                
                if subject and obj:
                    relationships.append({
                        'subject': subject,
                        'relation': token.text,
                        'object': obj
                    })
        
        return relationships
    
    def extract_fashion_relationships(self, text):
        """
        Extract fashion-specific relationships.
        Examples: "I want a red dress", "looking for Gucci bags"
        """
        doc = self.nlp(text.lower())
        relationships = []
        
        # Pattern: want/need/looking + product
        want_patterns = ['want', 'need', 'looking', 'searching', 'buy', 'purchase']
        
        for token in doc:
            if token.lemma_ in want_patterns:
                # Find what they want
                for child in token.children:
                    if child.dep_ in ['dobj', 'prep']:
                        # Get the full phrase
                        phrase = ' '.join([t.text for t in child.subtree])
                        relationships.append({
                            'intent': token.text,
                            'target': phrase
                        })
        
        return relationships

# ==================== Sentiment Analysis ====================

class SentimentAnalyzer:
    """
    Sentiment analysis using SpaCy and TextBlob patterns.
    Useful for understanding customer satisfaction.
    """
    
    def __init__(self):
        try:
            from textblob import TextBlob
            self.textblob_available = True
        except:
            self.textblob_available = False
            print("Note: TextBlob not available. Install with: pip install textblob")
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text.
        Returns polarity (-1 to 1) and subjectivity (0 to 1).
        """
        if self.textblob_available:
            from textblob import TextBlob
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'sentiment': 'positive' if blob.sentiment.polarity > 0 else 'negative' if blob.sentiment.polarity < 0 else 'neutral'
            }
        else:
            # Simple rule-based fallback
            positive_words = ['good', 'great', 'excellent', 'love', 'amazing', 'perfect', 'beautiful']
            negative_words = ['bad', 'terrible', 'hate', 'awful', 'poor', 'disappointing']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiment = 'positive'
                polarity = 0.5
            elif neg_count > pos_count:
                sentiment = 'negative'
                polarity = -0.5
            else:
                sentiment = 'neutral'
                polarity = 0.0
            
            return {
                'polarity': polarity,
                'sentiment': sentiment
            }

# ==================== Initialize Global Instances ====================

# Singleton instances for efficiency
_ner = None
_keyphrase_extractor = None
_relationship_extractor = None
_sentiment_analyzer = None

def get_ner():
    """Get or create NER instance."""
    global _ner
    if _ner is None:
        _ner = AdvancedNER()
    return _ner

def get_keyphrase_extractor():
    """Get or create keyphrase extractor."""
    global _keyphrase_extractor
    if _keyphrase_extractor is None:
        _keyphrase_extractor = KeyphraseExtractor()
    return _keyphrase_extractor

def get_relationship_extractor():
    """Get or create relationship extractor."""
    global _relationship_extractor
    if _relationship_extractor is None:
        _relationship_extractor = RelationshipExtractor()
    return _relationship_extractor

def get_sentiment_analyzer():
    """Get or create sentiment analyzer."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer

# ==================== Comprehensive Analysis Function ====================

def analyze_text_comprehensive(text):
    """
    Perform comprehensive NLP analysis on text.
    Combines all advanced NLP techniques.
    """
    results = {
        'original_text': text
    }
    
    # Named Entity Recognition
    ner = get_ner()
    results['entities'] = ner.extract_entities(text)
    results['fashion_entities'] = ner.extract_fashion_entities(text)
    results['noun_chunks'] = ner.get_noun_chunks(text)
    
    # Keyphrase Extraction
    try:
        kp_extractor = get_keyphrase_extractor()
        results['keyphrases'] = kp_extractor.extract_combined(text)
    except Exception as e:
        results['keyphrases'] = {'error': str(e)}
    
    # Relationship Extraction
    rel_extractor = get_relationship_extractor()
    results['relationships'] = rel_extractor.extract_relationships(text)
    results['fashion_relationships'] = rel_extractor.extract_fashion_relationships(text)
    
    # Sentiment Analysis
    sentiment = get_sentiment_analyzer()
    results['sentiment'] = sentiment.analyze_sentiment(text)
    
    return results

# ==================== Testing ====================

if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED NLP UTILITIES TEST")
    print("=" * 60 + "\n")
    
    test_text = "I'm looking for a beautiful Gucci handbag and maybe some Prada shoes for a wedding in Paris."
    
    print(f"Test Text: '{test_text}'\n")
    
    results = analyze_text_comprehensive(test_text)
    
    print("RESULTS:")
    print("=" * 60)
    
    print("\n1. Named Entities (SpaCy):")
    for ent in results['entities']:
        print(f"   • {ent['text']} ({ent['label']})")
    
    print("\n2. Fashion Entities:")
    print(f"   • Brands: {results['fashion_entities']['brands']}")
    print(f"   • Products: {results['fashion_entities']['products']}")
    
    print("\n3. Noun Chunks:")
    for chunk in results['noun_chunks']:
        print(f"   • {chunk}")
    
    print("\n4. Keyphrases:")
    if 'keybert' in results['keyphrases']:
        print("   KeyBERT:")
        for kp in results['keyphrases']['keybert']:
            print(f"      • {kp['phrase']} (score: {kp['score']:.3f})")
    
    print("\n5. Relationships:")
    for rel in results['relationships']:
        print(f"   • {rel}")
    
    print("\n6. Fashion Relationships:")
    for rel in results['fashion_relationships']:
        print(f"   • {rel}")
    
    print("\n7. Sentiment:")
    print(f"   • {results['sentiment']}")
    
    print("\n" + "=" * 60)
