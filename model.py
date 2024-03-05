import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from textblob import TextBlob
from textblob import Word

# Add the following line at the beginning of the file:
vectorizer = CountVectorizer()

df = pd.read_csv(r"data/Mental-Health-Twitter.csv")
df = df[['post_text']]

# Change all characters in tweets to lower case
df["post_text"] = df["post_text"].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Remove numbers from tweets
df["post_text"] = df["post_text"].str.replace("\d","")

# Remove punctuation from tweets
df["post_text"] = df["post_text"].str.replace("[^\w\s]","")

nltk.download("stopwords")

sw = stopwords.words("english")
df["post_text"] = df["post_text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

nltk.download("wordnet")
nltk.download("omw-1.4")
df["post_text"] = df["post_text"].apply(lambda x: " ".join([Word(x).lemmatize()]))

nltk.download('punkt')
from nltk.tokenize import word_tokenize
df["tokens"] = df["post_text"].apply(lambda x: TextBlob(x).words)

blob_emptylist = []

for i in df["post_text"]:
    blob = TextBlob(i).sentiment  # returns polarity
    blob_emptylist.append(blob)

df2 = pd.DataFrame(blob_emptylist)
df3 = pd.concat([df.reset_index(drop=True), df2], axis=1)
df4 = df3[['post_text', 'tokens', 'polarity']]

df4["Sentiment"] = np.where(df4["polarity"] >= 0, "Positive", "Negative")

# split the data into test and train set
X_train, X_test, y_train, y_test = train_test_split(df4['post_text'], df4['Sentiment'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(clf, 'sentimemt_plot.sav")
