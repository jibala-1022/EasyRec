import streamlit as st
import pandas as pd
import torch
import yaml
from embeddings import load_model, compute_embeddings

# Load configuration from YAML file
with open("configs.yaml", "r") as file:
    configs = yaml.safe_load(file)

# Load the processed movie dataset
processed_data_file = f"{configs['production_data']}/{configs['processed_dataset']}"
movie_data = pd.read_csv(processed_data_file)

# Streamlit app
st.title("🎬 EasyRec Movie Recommender")

# Dropdown for model selection
model_names = configs['hf_models']  # Assuming this is a list of model names in your configs.yaml
selected_model_name = st.selectbox("Select a model:", model_names)

# Load the model based on user selection
model, tokenizer = load_model(selected_model_name)

# User input for movie description
user_description = st.text_input("Enter a description of the type of movie you're interested in:", 
                                 placeholder="e.g. A romantic comedy with a twist...")

if user_description:
    # Load the precomputed movie embeddings from a .pt file
    embedding_dir_path = f"{configs['production_data']}/{configs['movie_embeddings']}/{selected_model_name}"
    embedding_file_path = f"{embedding_dir_path}/{configs['movie_embeddings']}.pt"
    movie_embeddings = torch.load(embedding_file_path)  # Load the .pt file

    # Compute the embedding for the user input by passing it as a list
    user_embedding = compute_embeddings([user_description], model, tokenizer)

    similarity_scores = torch.matmul(movie_embeddings, user_embedding.T).flatten()

    # Set the number of top recommendations to display
    K = 5  
    top_k_indices = torch.argsort(similarity_scores, descending=True)[:K].tolist()  # Get indices of top K

    # Display recommendations
    st.write("## 🎉 Top Recommendations:")
    
    for rank, movie_id in enumerate(top_k_indices, start=1):
        movie = movie_data.iloc[movie_id]
        
        # Convert runtime from minutes to hours and minutes
        hours = movie.runtime // 60
        minutes = movie.runtime % 60

        # Construct an HTML card for displaying the movie information
        st.markdown(f"### {rank}. {movie.title}")
        st.markdown(f"**Release Date:** {movie.release_date}  &nbsp;&nbsp; **Runtime:** {f'{hours}h {minutes}m' if hours > 0 else f'{minutes}m'}")
        st.markdown(f"⭐ {movie.vote_average} ({movie.vote_count} votes)")
        st.markdown(f"**Overview:** {movie.overview}")
        st.markdown(f"**Genres:** {movie.genres}")
        st.markdown(f"**Production Companies:** {movie.production_companies}")
        st.markdown(f"**Production Countries:** {movie.production_countries}")
        st.markdown("---")

# Additional styling (optional)
st.markdown(
    """
    <style>
    .stTextInput > div > input {
        background-color: #f0f0f5;  /* Light gray background */
        border: 1px solid #ccc;      /* Gray border */
        border-radius: 5px;          /* Rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True
)
