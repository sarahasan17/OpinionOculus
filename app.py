import streamlit as st
import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader
from sklearn.feature_extraction.text import CountVectorizer
from prediction import predict
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from string import punctuation
import nltk

st.title('YouTube Comment Sentiment Analysis')

# User input for YouTube link
youtube_link = st.text_input('Enter YouTube link:', 'https://www.youtube.com/watch?v=KNiEkPwH9S8')

if st.button('Fetch Comments and Analyze Sentiment'):
    # Extract video ID from the YouTube link
    video_id_match = re.search(r'v=([a-zA-Z0-9_-]+)', youtube_link)
    
    if video_id_match:
        video_id = video_id_match.group(1)
        
        # Initialize the YoutubeCommentDownloader object
        youtube = YoutubeCommentDownloader()
        
        # Fetch video comments by passing the video ID to get_comments()
        response = youtube.get_comments(youtube_id=video_id)

        all_data = []
        for i, comment in enumerate(response):
            all_data.append(comment)
            if i >= 100:  # It will print the structure of the first 100 comments
                break
        
        # Create DataFrame from comments
        df = pd.DataFrame(all_data)
        
        # Process sentiment analysis
        processed_data = process_sentiment_analysis(df)
        
        # Display the DataFrame with sentiment analysis results
        st.write(processed_data)
        
    else:
        st.error("Invalid YouTube link. Please provide a valid link.")

# Function to process sentiment analysis
def process_sentiment_analysis(data):
    # Process data
    data1 = data.drop(columns=['Unnamed: 0', 'likes', 'time', 'user', 'userlink'], axis=1)

    # Read the CSV file into a DataFrame
    data1 = pd.read_csv(data1)

    # Check the columns in the DataFrame
    st.write(data1.columns)

    # Create a new DataFrame (data1) by dropping specified columns
    columns_to_drop = ['Unnamed: 0', 'likes', 'time', 'user', 'userlink']
    existing_columns = list(data1.columns)
    columns_to_drop = [col for col in columns_to_drop if col in existing_columns]  # Filter out non-existent columns

    data1 = data1.drop(columns=columns_to_drop, axis=1)

    # Check the columns in the DataFrame
    st.write(data1.columns)

    # Create a new DataFrame (data1) by dropping specified columns
    columns_to_drop = ['Unnamed: 0', 'likes', 'time', 'user', 'userlink']
    existing_columns = list(data1.columns)
    columns_to_drop = [col for col in columns_to_drop if col in existing_columns]  # Filter out non-existent columns

    data1 = data1.drop(columns=columns_to_drop, axis=1)

    # Perform data preprocessing & preparation
    nltk.download('vader_lexicon')
    sentiments = SentimentIntensityAnalyzer()

    data1["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data1["text"]]
    data1["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data1["text"]]
    data1["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data1["text"]]
    data1['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in data1["text"]]
    score = data1["Compound"].values
    sentiment = []
    for i in score:
        if i >= 0.05:
            sentiment.append('Positive')
        elif i <= -0.05:
            sentiment.append('Negative')
        else:
            sentiment.append('Neutral')
    data1["Sentiment"] = sentiment

    # ... (rest of the processing code)

    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    porter_stemmer = PorterStemmer()
    lancaster_stemmer = LancasterStemmer()
    snowball_stemer = SnowballStemmer(language="english")
    lzr = WordNetLemmatizer()

    # Text processing using nltk
    data_copy = data1.copy()
    data_copy.text = data_copy.text.apply(lambda text: text_processing(text))

    le = LabelEncoder()
    data_copy['Sentiment'] = le.fit_transform(data_copy['Sentiment'])

    processed_data = {
        'Sentence': data_copy.text,
        'Sentiment': data_copy['Sentiment']
    }

    processed_data = pd.DataFrame(processed_data)
    st.write(processed_data.head())

    st.write(processed_data['Sentiment'].value_counts())

    # Data visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    sentiment_counts = processed_data['Sentiment'].value_counts()
    colors = sns.color_palette("pastel", len(sentiment_counts))

    ax.bar(sentiment_counts.index.map({0: 'Negative', 1: 'Neutral', 2: 'Positive'}), sentiment_counts.values, color=colors)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Comments')
    ax.set_title('Distribution of Sentiment Classes after Upsampling')

    plt.savefig('sentiment_distribution.png')

    with open('sentiment_plot.sav', 'wb') as f:
        pickle.dump(fig, f)

    return processed_data

def text_processing(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub('[%s]' % re.escape(punctuation), "", text)
    text = re.sub("^a-zA-Z0-9$,.", "", text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'\W', ' ', text)
    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])

    text = ' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])

    return text
