import json
import os

import pandas as pd
import torch
import yaml

from embeddings import compute_embeddings, load_model

# Load configurations
with open("configs.yaml", "r") as file:
    configs = yaml.safe_load(file)

# Load and process the movie dataset
data_file = f"{configs['production_data']}/{configs['dataset']}"
movies_data = pd.read_csv(data_file)

# Define columns to drop that are not needed
columns_drop = ['budget', 'homepage', 'id', 'original_language', 'original_title', 
                'popularity', 'revenue', 'spoken_languages', 'status', 'tagline']
movies_data.drop(columns=columns_drop, axis=1, inplace=True)
movies_data.dropna(inplace=True)  # Drop rows with missing values

# Convert JSON string columns to a comma-separated string of names
columns_json_to_csv = ['genres', 'keywords', 'production_companies', 'production_countries']
for col in columns_json_to_csv:
    movies_data[col] = movies_data[col].apply(
        lambda json_str: ', '.join([item["name"] for item in json.loads(json_str)])
    )

# Extract the year from 'release_date'
movies_data['release_date'] = pd.to_datetime(movies_data['release_date']).dt.year

# Convert 'runtime' to integers
movies_data['runtime'] = movies_data['runtime'].astype(int)

# Combine 'overview', 'genres', and 'keywords' into a single string for each movie
movies_data_processed = movies_data[['overview', 'genres', 'keywords']].apply(
    lambda row: '. '.join([f"{col.capitalize()}: {val}" for col, val in row.items()]), 
    axis=1
).tolist()

# Save the processed dataset
processed_data_file = f"{configs['production_data']}/{configs['processed_dataset']}"
movies_data.to_csv(processed_data_file, index=False)

# Process embeddings for each model
for model_name in configs['hf_models']:
    model, tokenizer = load_model(model_name)
    movie_embeddings = compute_embeddings(movies_data_processed, model, tokenizer)
    
    embedding_dir_path = f"{configs['production_data']}/{configs['movie_embeddings']}/{model_name}"
    embedding_file_path = f"{embedding_dir_path}/{configs['movie_embeddings']}.pt"
    os.makedirs(embedding_dir_path, exist_ok=True)

    torch.save(movie_embeddings, embedding_file_path)
    print(f"Saved embeddings for {model_name}")

