import pandas as pd

# Load the datasets
movies = pd.read_csv('movies.csv')  # Replace with your file path
ratings = pd.read_csv('ratings.csv')  # Replace with your file path

# Display first few rows of each dataset
print(movies.head())
print(ratings.head())
