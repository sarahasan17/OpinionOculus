# Import necessary libraries
pip install -r requirements.txt

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from youtube_comment_downloader import YoutubeCommentDownloader
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string

# Function to process text data
def text_processing(text):
    # Include your text processing logic here
    # For example, lowercasing, removing punctuation, stopwords, and lemmatization
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub("^a-zA-Z0-9$,.", "", text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'\W', ' ', text)
    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    text = ' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])
    return text

# Function to load saved model or any other saved object
def load_model():
    # Load your saved model or object using pickle
    with open('sentiment_plot.sav', 'rb') as f:
        model = pickle.load(f)
    return model

# Streamlit App
def main():
    st.title("YouTube Sentiment Analysis App")

    # User input for YouTube link
    link = st.text_input("Enter YouTube link:")
    
    # Check if the link is provided
    if link:
        st.subheader("Extracting Comments...")
        
        # Extract video ID from the YouTube link
        video_id_match = re.search(r'v=([a-zA-Z0-9_-]+)', link)
        if video_id_match:
            video_id = video_id_match.group(1)
        else:
            st.error("Invalid YouTube link. Please provide a valid link.")
            st.stop()

        # Initialize the YoutubeCommentDownloader object
        youtube = YoutubeCommentDownloader()

        # Fetch video comments by passing the video ID to get_comments()
        response = youtube.get_comments(youtube_id=video_id)

        # Display the structure of the first 100 comments
        st.write(pd.DataFrame(response[:100]))

        st.subheader("Processing Comments...")

        # Process comments using the text_processing function
        processed_comments = [text_processing(comment) for comment in response]

        # Display processed comments
        st.write(pd.DataFrame(processed_comments, columns=['Processed Comments']))

        st.subheader("Sentiment Analysis Results")

        # Include your sentiment analysis results visualization here
        # For example, creating a bar plot of sentiment distribution
        sentiment_counts = pd.Series(sentiment).value_counts()
        st.bar_chart(sentiment_counts)

        st.subheader("Distribution of Sentiment Classes")

        # Load and display the saved sentiment distribution plot
        model = load_model()
        st.pyplot(model)

if __name__ == "__main__":
    main()
