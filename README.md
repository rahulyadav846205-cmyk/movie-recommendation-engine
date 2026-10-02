# Movie Recommendation Engine

## Task 4 - CODTECH Internship

This project is a **content-based movie recommendation system**. It recommends movies similar to a movie selected by the user.

### Technologies
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity

### Project Structure
```text
movie-recommendation-engine/
├── movies.csv
├── movie_recommender.py
├── requirements.txt
└── README.md
```

### How It Works
1. Loads the movie dataset.
2. Combines each movie's genres and description into one text field.
3. Converts the text into TF-IDF vectors.
4. Calculates cosine similarity between movies.
5. Finds movies most similar to the selected title.
6. Displays the top 5 recommendations.

### Run the Project
```bash
pip install -r requirements.txt
python movie_recommender.py
```

### Example
If the user enters:
```text
Inception
```

the engine returns movies with similar action/science-fiction/thriller content.

### Note
The included dataset is a compact demonstration dataset created for the internship task. A production recommendation system should use a larger, validated movie catalogue and may combine content-based and collaborative filtering techniques.
