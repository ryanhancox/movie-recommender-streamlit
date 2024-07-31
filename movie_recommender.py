# Imports
import pandas as pd
from sentence_transformers import SentenceTransformer
import ast

transformer = SentenceTransformer('all-MiniLM-L6-v2')
    
# Function to load and clean movie data
def prepare_data() -> pd.DataFrame:
    df_movies_metadata = pd.read_csv(r'./data/movies_metadata.csv', low_memory=False)
    df_movies = df_movies_metadata[['title', 'overview', 'release_date', 'genres']]
    df_movies_cleaned = df_movies.dropna(how='any').reset_index()
    return df_movies_cleaned


# Function to embed search string
def embed_search(search_string):
    search_embeddings = transformer.encode(search_string)
    return search_embeddings


# Function that performs similarity search using the embedded search string
def search_for_similar_movies(client, search_string):
    K = 5
    SEARCH_PARAM = {
        "metric_type": "L2",
        "params": {"nprobe": 20}
    }
    search_vector = embed_search(search_string)
    results = client.search("movie_vectors", [search_vector], search_params=SEARCH_PARAM,
                            limit=K)
    return results

# Build genres for movie
def build_genres(row):
    genre_list = []
    entries = ast.literal_eval(row['genres'])
    for entry in entries:
        genre_list.append(entry['name'])
    genres = ', '.join(genre_list)
    return genres
