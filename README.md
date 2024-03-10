# OPINIONOCULUS
## Youtube-comment-sentimental-analysis
## 1. YouTube Comment Retrieval

    Extract comments from a specified YouTube video using the youtube-comment-downloader library.
    The video link and the output file name are obtained through user input.
    The script fetches the comments and stores them in a Pandas DataFrame.

## 2. Data Preprocessing

    Load the initial DataFrame and drop unnecessary columns.
    Perform sentiment analysis using the VADER sentiment analysis tool from the NLTK library.
    Calculate sentiment scores (positive, negative, neutral, and compound) for each comment.
    Create a new column "Sentiment" based on the compound score.

## 3. Text Preprocessing

    Tokenize the comments using NLTK's word tokenizer.
    Convert text to lowercase to ensure uniformity.
    Remove punctuation and special characters.
    Remove stop words using NLTK's predefined stop word list.
    Lemmatize the words to reduce them to their base or root form.

## 4. Data Balancing

    Balance the dataset by upsampling the minority classes (negative and neutral sentiments) to match the majority class (positive sentiment).

## 5. Visualization

    Plot a bar graph to visualize the distribution of sentiment classes in the balanced dataset.

## 6. Results

    Display positive comments from the dataset for further analysis.

The emphasis here is on the text preprocessing steps and the use of NLP techniques, such as sentiment analysis, tokenization, lemmatization, and handling class imbalance. These steps are crucial for preparing the text data before feeding it into any machine learning model.
