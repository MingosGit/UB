"""
@author: Jose Candon y Pau Gonzalez
Per utilitzar aquest mòdul en un notebook:
    import practica3 as pt3

Exemple d'ús complet:
    # Importar el mòdul
    import ngrams_classifier as pt3
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Carregar dades
    df = pd.read_csv('dataset.csv')
    texts = df['Text'].tolist()
    labels = df['language'].tolist()
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    # Extreure n-grames
    X_train_ngrams = pt3.extract_ngrams_from_texts(X_train, n=2)
    X_test_ngrams = pt3.extract_ngrams_from_texts(X_test, n=2)
    
    # Vectoritzar amb TF-IDF
    vectorizer = TfidfVectorizer(analyzer=lambda x: x)
    X_train_tfidf = vectorizer.fit_transform(X_train_ngrams)
    X_test_tfidf = vectorizer.transform(X_test_ngrams)
    
    # Entrenar i avaluar
    model = pt3.MultinomialNBfit(X_train_tfidf, y_train)
    accuracy = pt3.MultinomialNBpredict(model, X_test_tfidf, y_test)
    print(f"Accuracy: {accuracy:.4f}")
"""

from itertools import combinations
import re
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score


def get_open_ngrams(word, n, include_boundaries=True):
    """
    Extrae n-gramas oberts d'una paraula.
    
    Args:
        word: paraula a processar
        n: ordre de l'n-gram (2 per bigrames, 3 per trigrames, etc.)
        include_boundaries: si es True, afegeix '_' al principi i final
    
    Returns:
        set amb els n-grames oberts
    """
    open_ngrams = set()
    
    if include_boundaries:
        extended_word = '_' + word + '_'
        last_idx = len(extended_word) - 1
        
        for combo in combinations(range(len(extended_word)), n):
            if 0 in combo and 1 not in combo:
                continue
            if last_idx in combo and (last_idx - 1) not in combo:
                continue
            
            ngram = ''.join(extended_word[i] for i in combo)
            open_ngrams.add(ngram)
    else:
        for combo in combinations(range(len(word)), n):
            ngram = ''.join(word[i] for i in combo)
            open_ngrams.add(ngram)
    
    return open_ngrams


def analyze_collisions(texts, n=2):
    """
    Analitza col·lisions entre paraules basant-se en els seus n-grames.
    
    Args:
        texts: llista de textos a analitzar
        n: ordre dels n-grames (default: 2)
    
    Returns:
        diccionari amb estadístiques de col·lisions
    """
    unique_words = set()
    for text in texts:
        text_lower = text.lower()
        words_in_text = re.findall(r'\b\w+\b', text_lower)
        unique_words.update(words_in_text)
    
    word_to_ngrams = {}
    ngrams_to_words = defaultdict(list)
    
    for word in unique_words:
        ngrams = frozenset(get_open_ngrams(word, n, include_boundaries=True))
        word_to_ngrams[word] = ngrams
        ngrams_to_words[ngrams].append(word)
    
    colliding_words = set()
    collision_examples = []
    
    for ngrams, words in ngrams_to_words.items():
        if len(words) > 1:
            colliding_words.update(words)
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    collision_examples.append((words[i], words[j]))
    
    return {
        'unique_words': len(unique_words),
        'unique_ngram_sets': len(ngrams_to_words),
        'colliding_words': len(colliding_words),
        'collision_examples': collision_examples[:10]
    }


def extract_ngrams_from_texts(texts, n=2):
    """
    Extreu els n-grames de cada text.
    
    Args:
        texts: llista de textos
        n: ordre dels n-grames (default: 2)
    
    Returns:
        llista de llistes amb els n-grames de cada text
    """
    texts_ngrams = []
    
    for text in texts:
        text_lower = text.lower()
        words_in_text = re.findall(r'\b\w+\b', text_lower)
        
        text_ngrams = []
        for word in words_in_text:
            word_ngrams = get_open_ngrams(word, n, include_boundaries=True)
            text_ngrams.extend(word_ngrams)
        
        texts_ngrams.append(text_ngrams)
    
    return texts_ngrams


def MultinomialNBfit(X, y):
    """
    Entrena un classificador Naive Bayes multinomial.
    
    Args:
        X: matriu sparse amb les característiques (TF-IDF o freqüències)
        y: llista amb les etiquetes de classe
    
    Returns:
        diccionari amb el model entrenat
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n_features = X.shape[1]
    
    class_log_prior = np.zeros(n_classes)
    feature_log_prob = np.zeros((n_classes, n_features))
    
    for idx, cls in enumerate(classes):
        mask = np.array([label == cls for label in y])
        X_cls = X[mask]
        
        class_log_prior[idx] = np.log(X_cls.shape[0] / X.shape[0])
        
        feature_count = np.asarray(X_cls.sum(axis=0)).ravel()
        
        smoothed_count = feature_count + 1
        smoothed_total = smoothed_count.sum()
        
        feature_log_prob[idx, :] = np.log(smoothed_count / smoothed_total)
    
    model = {
        'classes': classes,
        'class_log_prior': class_log_prior,
        'feature_log_prob': feature_log_prob
    }
    
    return model


def MultinomialNBpredict(model, X, y=None):
    """
    Fa prediccions amb el classificador Naive Bayes.
    
    Args:
        model: model entrenat amb MultinomialNBfit
        X: matriu sparse amb les característiques a classificar
        y: etiquetes reals (opcional, només per calcular accuracy)
    
    Returns:
        si y és None, retorna les prediccions
        si y es proporciona, retorna l'accuracy
    """
    classes = model['classes']
    class_log_prior = model['class_log_prior']
    feature_log_prob = model['feature_log_prob']
    
    log_prob = X.dot(feature_log_prob.T) + class_log_prior
    
    y_pred = classes[np.argmax(log_prob, axis=1)]
    
    if y is not None:
        return accuracy_score(y, y_pred)
    else:
        return y_pred
