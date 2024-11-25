from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load movie data
movies = pd.read_csv('movies.csv')  # Replace with your file path

# Check the columns in the dataset
print("Columns in movies dataset:", movies.columns)

# Check the first few movie titles to understand the structure
print("First 10 movie titles:", movies['title'].head(10))

# Using the 'genres' column for movie similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])  # You can modify this to 'description' if you have that

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on similarity
def recommend_movie(movie_title):
    # Search case-insensitively for the movie title in the dataset
    matched_movies = movies[movies['title'].str.contains(movie_title, case=False)]
    
    print(f"Matched movies for '{movie_title}':", matched_movies['title'].values)  # Debugging line
    
    if matched_movies.empty:
        return f"Sorry, no movies found for '{movie_title}'."
    
    # Assuming we take the first match found
    idx = matched_movies.index[0]

    # Get pairwise similarity scores for all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies based on similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 5 most similar movies
    movie_indices = [i[0] for i in sim_scores[1:6]]

    return movies['title'].iloc[movie_indices]

# Example usage and testing
print(recommend_movie("The Matrix"))  # Test with the movie title
print(movies['title'].head(10))  # Print the first 10 movie titles

# Check if 'Matrix' exists in any movie titles (case-insensitive search)
print(movies[movies['title'].str.contains('Matrix', case=False)])  # Case-insensitive search
