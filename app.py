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
      "source": [
        "!pip install --upgrade pip\n",
        "!pip install youtube-comment-downloader\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4XqKNxpHa8xK",
        "outputId": "92fdcf88-7971-474e-cf78-187fe70c3ac1"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting youtube-comment-downloader\n",
            "  Downloading youtube_comment_downloader-0.1.70-py3-none-any.whl (7.9 kB)\n",
            "Collecting dateparser (from youtube-comment-downloader)\n",
            "  Downloading dateparser-1.2.0-py2.py3-none-any.whl (294 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m295.0/295.0 kB\u001b[0m \u001b[31m6.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from youtube-comment-downloader) (2.31.0)\n",
            "Requirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from dateparser->youtube-comment-downloader) (2.8.2)\n",
            "Requirement already satisfied: pytz in /usr/local/lib/python3.10/dist-packages (from dateparser->youtube-comment-downloader) (2023.4)\n",
            "Requirement already satisfied: regex!=2019.02.19,!=2021.8.27 in /usr/local/lib/python3.10/dist-packages (from dateparser->youtube-comment-downloader) (2023.12.25)\n",
            "Requirement already satisfied: tzlocal in /usr/local/lib/python3.10/dist-packages (from dateparser->youtube-comment-downloader) (5.2)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->youtube-comment-downloader) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->youtube-comment-downloader) (3.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->youtube-comment-downloader) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->youtube-comment-downloader) (2024.2.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil->dateparser->youtube-comment-downloader) (1.16.0)\n",
            "Installing collected packages: dateparser, youtube-comment-downloader\n",
            "Successfully installed dateparser-1.2.0 youtube-comment-downloader-0.1.70\n"
          ]
        }
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
      "execution_count": null,
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
      "authorship_tag": "ABX9TyP/1wUzX0kubesSP1govwlY",
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