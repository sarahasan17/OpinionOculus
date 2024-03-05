{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/sarahasan17/Youtube-comment-sentimental-analysis/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 390
        },
        "id": "b3y82QgYHnXY",
        "outputId": "d88678bb-6610-4179-a470-7b27a939639d"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "ModuleNotFoundError",
          "evalue": "No module named 'streamlit'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-4-7963525d8e71>\u001b[0m in \u001b[0;36m<cell line: 2>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# Import necessary libraries\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0;32mimport\u001b[0m \u001b[0mstreamlit\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mst\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      3\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mpandas\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mmatplotlib\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpyplot\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mseaborn\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0msns\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'streamlit'",
            "",
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0;32m\nNOTE: If your import is failing due to a missing package, you can\nmanually install dependencies using either !pip or !apt.\n\nTo view examples of installing some common dependencies, click the\n\"Open Examples\" button below.\n\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"
          ],
          "errorDetails": {
            "actions": [
              {
                "action": "open_url",
                "actionText": "Open Examples",
                "url": "/notebooks/snippets/importing_libraries.ipynb"
              }
            ]
          }
        }
      ],
      "source": [
        "# Import necessary libraries\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import pickle\n",
        "from youtube_comment_downloader import YoutubeCommentDownloader\n",
        "from nltk.sentiment.vader import SentimentIntensityAnalyzer\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "from nltk.tokenize import word_tokenize\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "from nltk.corpus import stopwords\n",
        "import re\n",
        "import string\n",
        "\n",
        "# Function to process text data\n",
        "def text_processing(text):\n",
        "    # Include your text processing logic here\n",
        "    # For example, lowercasing, removing punctuation, stopwords, and lemmatization\n",
        "    text = text.lower()\n",
        "    text = re.sub(r'\\n', ' ', text)\n",
        "    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)\n",
        "    text = re.sub(\"^a-zA-Z0-9$,.\", \"\", text)\n",
        "    text = re.sub(r'\\s+', ' ', text, flags=re.I)\n",
        "    text = re.sub(r'\\W', ' ', text)\n",
        "    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])\n",
        "    text = ' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])\n",
        "    return text\n",
        "\n",
        "# Function to load saved model or any other saved object\n",
        "def load_model():\n",
        "    # Load your saved model or object using pickle\n",
        "    with open('sentiment_plot.sav', 'rb') as f:\n",
        "        model = pickle.load(f)\n",
        "    return model\n",
        "\n",
        "# Streamlit App\n",
        "def main():\n",
        "    st.title(\"YouTube Sentiment Analysis App\")\n",
        "\n",
        "    # User input for YouTube link\n",
        "    link = st.text_input(\"Enter YouTube link:\")\n",
        "\n",
        "    # Check if the link is provided\n",
        "    if link:\n",
        "        st.subheader(\"Extracting Comments...\")\n",
        "\n",
        "        # Extract video ID from the YouTube link\n",
        "        video_id_match = re.search(r'v=([a-zA-Z0-9_-]+)', link)\n",
        "        if video_id_match:\n",
        "            video_id = video_id_match.group(1)\n",
        "        else:\n",
        "            st.error(\"Invalid YouTube link. Please provide a valid link.\")\n",
        "            st.stop()\n",
        "\n",
        "        # Initialize the YoutubeCommentDownloader object\n",
        "        youtube = YoutubeCommentDownloader()\n",
        "\n",
        "        # Fetch video comments by passing the video ID to get_comments()\n",
        "        response = youtube.get_comments(youtube_id=video_id)\n",
        "\n",
        "        # Display the structure of the first 100 comments\n",
        "        st.write(pd.DataFrame(response[:100]))\n",
        "\n",
        "        st.subheader(\"Processing Comments...\")\n",
        "\n",
        "        # Process comments using the text_processing function\n",
        "        processed_comments = [text_processing(comment) for comment in response]\n",
        "\n",
        "        # Display processed comments\n",
        "        st.write(pd.DataFrame(processed_comments, columns=['Processed Comments']))\n",
        "\n",
        "        st.subheader(\"Sentiment Analysis Results\")\n",
        "\n",
        "        # Include your sentiment analysis results visualization here\n",
        "        # For example, creating a bar plot of sentiment distribution\n",
        "        sentiment_counts = pd.Series(sentiment).value_counts()\n",
        "        st.bar_chart(sentiment_counts)\n",
        "\n",
        "        st.subheader(\"Distribution of Sentiment Classes\")\n",
        "\n",
        "        # Load and display the saved sentiment distribution plot\n",
        "        model = load_model()\n",
        "        st.pyplot(model)\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "PA2yK_Wga6Z_"
      },
      "execution_count": None,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "spHXwmahxzov"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMhuTRZBDARkbZe96R+wGJV",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
