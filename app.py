from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load movie data
movies = pd.read_csv('movies.csv')  # Replace with your file path

# Assuming 'movies' has a 'description' or 'genres' column (you can modify it)
# Create a TF-IDF Vectorizer and compute cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Use the 'genres' column for recommendation (can change to 'description' if you have it)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])  # Or 'description' if you have that
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on similarity
def recommend_movie(movie_title):
    # Get index of the movie that matches the title
    idx = movies[movies['title'].str.contains(movie_title, case=False)].index

    if len(idx) == 0:
        return []  # If no movie is found, return an empty list

    idx = idx[0]  # Use the first match

    # Get pairwise similarity scores for all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 5 most similar movies
    movie_indices = [i[0] for i in sim_scores[1:6]]

    # Return the movie titles as a list
    return movies['title'].iloc[movie_indices].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie_title']  # Get movie title from form

    # Generate recommendations
    recommended_movies = recommend_movie(movie_title)

    # Ensure `recommended_movies` is a list of strings (titles)
    if isinstance(recommended_movies, pd.Series):
        recommended_movies = recommended_movies.tolist()

    # Render the template with recommended movies
    return render_template('recommendations.html', recommended_movies=recommended_movies, movie_title=movie_title)

if __name__ == '__main__':
    app.run(debug=True)
