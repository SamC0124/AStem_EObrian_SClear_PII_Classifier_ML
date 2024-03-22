# Imports
import nltk
from nltk.corpus import stopwords
import numpy as np
import os
import pandas as pd
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from string import punctuation

# Load testing data
raw_data = pd.read_json(f"data/train.json")
raw_data.head()
print(raw_data['tokens'][0])

# Create a set of punctuation and special characters to omit from tokens from punctuation library, and other tokens that we found randomly in the list
omitted_characters = set(punctuation)
omitted_characters.add("\n\n")
omitted_characters.add("\n")
omitted_characters.add("\r\n")
omitted_characters.add("\r")
omitted_characters.add(" ")
omitted_characters.add("")
omitted_characters.add("  ")
omitted_characters.add("•")
omitted_characters.add("–")
omitted_characters.add(',')

def clean_tokens(data):
    for row, tokens in enumerate(data['tokens']):
        for token_index, token in enumerate(tokens):
            token_stripped = token.strip()
            token_lower = token_stripped.lower()
            if token_lower in omitted_characters:
                # Remove token from token list.
                del data['tokens'][row][token_index]
            elif token.isalpha() == False:
                # Remove token from token list.
                del data['tokens'][row][token_index]
            else:
                data['tokens'][row][token_index] = token_lower

            # TODO: Remove all words that contain an omitted character? No, this breaks hyphenated words

    return data

def remove_stopwords(data):
    nltk.download('stopwords')
    stop_words = set(list(stopwords.words('english')) + ['and', 'a'])
    for row, tokens in enumerate(data['tokens']):
        for token_index, word in enumerate(tokens):
            if word.lower() in stop_words:
                data['tokens'][row].pop(token_index)

    return data

# Splitting Documents by Segments (header sections and seperate sentences, in hopes of improving accuracy for the model
def seperate_segments(data):
    segments = []
    labels_by_segment = []
    labels = data['labels']
    within_parantheses = False
    new_segment = []
    new_labels = []
    for row, tokens in enumerate(data['tokens']):
        for token_idx, token in enumerate(tokens):
            if token == ")":
                within_parantheses = False
            elif token == "(":
                within_parantheses = True
            elif not within_parantheses and (token == "." or token == "\n\n" or token == "!" or token == "?"):
                if new_segment != []:
                    labels_by_segment.append(new_labels)
                    segments.append(new_segment)
                new_labels = []
                new_segment = []
            else:
                new_labels.append(data['labels'][row][token_idx])
                new_segment.append(token)

    # Return dictionary of new tokens and labels, allowing to skip removing the previous classifiers.
    return {"tokens": segments, "labels": labels_by_segment}

# Edit: Fixing shadowing issues that may be overwriting what is being done to clean the tokens
def clean_labels(data):
    cleaned_data = data
    for index, segment_labels in enumerate(data['labels']):
        cleaned_data.loc[index, 'labels'] = 0
        for label in segment_labels:
            if label != 'O':
                cleaned_data.loc[index, 'labels'] = 1

    return cleaned_data

cleaned_data = seperate_segments(raw_data)
cleaned_data = clean_tokens(cleaned_data)
cleaned_data = clean_tokens(cleaned_data)
cleaned_data = remove_stopwords(cleaned_data)
cleaned_data = remove_stopwords(cleaned_data)

print(f"Tokens from the given segment: {0}", cleaned_data['tokens'][0])
for idx in range(len(cleaned_data['tokens'])):
    if 'Avril' in cleaned_data['tokens'][idx]:
        print("here")

# TODO: Tokens with "," and "around" make it through these functions. This should be fixed, but fixes have been exhausted for now

def use_word2vec(data):
    X_wtv = data['tokens']
    model_current = Word2Vec(X_wtv, vector_size=100, window=5, min_count=5, workers=4)
    model_current.train(X_wtv, total_examples=len(X_wtv), epochs=10)
    return model_current

# Creating model for evaluating weights of words, based on labels around them?
model = use_word2vec(cleaned_data)

X = np.array([np.mean(model.wv[segment_idx], axis=0) for segment_idx in range(len(cleaned_data['tokens']))])
y = np.array(cleaned_data['labels'])
# Current Random-Forest Model Under Review
"""
# Split data into training and testing sets (adjust test_size and random_state as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Tokenization using CountVectorizer (you can also use TF-IDF vectorizer)
# X_train_vectorized, X_test_vectorized = create_bag_of_words(X_train, X_test)

# #Try SMOTE to fix class imbalance -- *** tried, made precision, recall, accuracy worse
# X_train_vectorized, y_train = use_smote(X_train_vectorized, y_train)

#Train RF classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(X_train_vectorized, y_train)

#Predict the labels for test data
y_pred = rf_classifier.predict(X_test_vectorized)

#Evaluate this classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print("Confusion Matrix:")
print(conf_matrix)
"""

# Sample Gaussian Naive-Bayes Model from SKLearn
# X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
exit()
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print("Confusion Matrix:")
print(conf_matrix)