import streamlit as st
from pymilvus import MilvusClient
from movie_recommender import (
    prepare_data, search_for_similar_movies, build_genres)

client = MilvusClient(uri="http://localhost:19530")
client.load_collection(collection_name="movie_vectors")
df_movies = prepare_data()

st.title("Movie Recommendation App")
user_txt = st.text_area(label= ("Enter description of a type of movie that you would like" 
                     " suggestions for:"), value="")

results = search_for_similar_movies(client, user_txt)

if user_txt:
    for entry in results[0]:
        expander = st.expander(f"**{df_movies['title'][entry['id']]}**")
        expander.write(df_movies['overview'][entry['id']])
        expander.write(f"*__Release date__: {df_movies['release_date'][entry['id']]}*")
        expander.write(f"Genres: {build_genres(df_movies.iloc[entry['id']])}")

        
