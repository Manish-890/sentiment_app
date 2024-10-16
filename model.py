# This file creates the 'pipe' NLP model and saves it as model.joblib
# Import libraries
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocessor

tfidf = TfidfVectorizer()
classifier = LinearSVC()

if __name__ == "__main__":
    # Change the following to your location of sentiments.csv
    try:
        df = pd.read_csv('sentiments.csv')  # Update path if necessary
        pipe = make_pipeline(preprocessor(), tfidf, classifier)
        pipe.fit(df['text'], df['sentiment'])
        joblib.dump(pipe, open('model.joblib', 'wb'))
        print("Model trained and saved as 'model.joblib'.")
    except FileNotFoundError:
        print("Error: The file 'sentiments.csv' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
