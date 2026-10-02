import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "movies.csv"

def build_recommender():
    df = pd.read_csv(DATA_FILE)
    df["content"] = (
        df["genres"].fillna("") + " " +
        df["description"].fillna("")
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(df["content"])
    similarity = cosine_similarity(matrix)

    indices = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()
    return df, similarity, indices

def recommend(title, df, similarity, indices, n=5):
    key = title.lower().strip()
    if key not in indices:
        return None

    idx = indices[key]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [x for x in scores if x[0] != idx][:n]

    result = df.iloc[[i for i, _ in scores]][["title","genres"]].copy()
    result["similarity_score"] = [round(score, 3) for _, score in scores]
    return result

def main():
    df, similarity, indices = build_recommender()

    print("=" * 60)
    print("MOVIE RECOMMENDATION ENGINE")
    print("=" * 60)
    print("\nAvailable movies:")
    for title in df["title"]:
        print("-", title)

    title = input("\nEnter a movie title: ").strip()
    result = recommend(title, df, similarity, indices, n=5)

    if result is None:
        print("\nMovie not found. Please enter a title exactly as shown.")
    else:
        print(f"\nRecommendations for '{title}':")
        print(result.to_string(index=False))

if __name__ == "__main__":
    main()
