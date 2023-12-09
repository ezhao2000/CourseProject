from bs4 import BeautifulSoup
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer

# IMDb dataset source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
df = pd.read_csv(r"C:\Users\ezhao\Downloads\IMDB_Dataset.csv")


# Function to clean/preprocess text
def clean_text(text):
    try:
        # Attempt to treat the text as HTML
        text = BeautifulSoup(text, 'html.parser').get_text()
    except Exception as e:
        pass

    # Convert text to lowercase
    text = text.lower()

    # Remove special characters/numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]

    # Join the words back into string
    cleaned_text = ' '.join(words)

    return cleaned_text


# Apply the cleaning function to the 'review' column
df['cleaned_review'] = df['review'].apply(clean_text)


# print(df[['review', 'cleaned_review', 'sentiment']].head())


nltk.download('stopwords')
nltk.download('wordnet')

# Tokenization and lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def tokenize_and_lemmatize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)


df['cleaned_review'] = df['cleaned_review'].apply(tokenize_and_lemmatize)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

# Train Naive Bayes classifier on entire dataset
classifier = MultinomialNB()
classifier.fit(X_tfidf, y)

# Cross-validation
cv_scores = cross_val_score(classifier, X_tfidf, y, cv=5, scoring='accuracy')
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean Cross-validation accuracy: {cv_scores.mean():.2%}')

# Classification report for entire dataset
y_pred_all = classifier.predict(X_tfidf)
print('\nClassification Report for the Entire Dataset:\n', classification_report(y, y_pred_all))

# User input loop
while True:
    # User input for sentiment analysis
    user_review = input("Enter a movie review (or type 'exit' to end): ")

    # Check if the user wants to exit
    if user_review.lower() == 'exit':
        break

    # Clean and preprocess the user input
    cleaned_user_review = tokenize_and_lemmatize(user_review)

    # TF-IDF Vectorization for user input
    user_review_tfidf = vectorizer.transform([cleaned_user_review])

    # Predict sentiment for user input
    user_review_sentiment = classifier.predict(user_review_tfidf)[0]

    # Map sentiment label to human-readable output
    sentiment_mapping = {'positive': 'Good', 'negative': 'Bad'}
    output_sentiment = sentiment_mapping.get(user_review_sentiment, 'Neutral')

    # Output the result based on user input
    print(f"The sentiment of the entered movie review is: {output_sentiment}")
