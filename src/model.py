import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_model(data_path: str):
    df = pd.read_csv(data_path)

    # Фильтруем только строки с известной difficulty
    df = df.dropna(subset=["difficulty"])

    X = df["question"]
    y = df["difficulty"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_vec, y)

    return model, vectorizer
