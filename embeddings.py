from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from model import Easyrec


def load_model(model_path: str) -> Tuple[Easyrec, AutoTokenizer]:
    """
    Load the pre-trained model and tokenizer from the specified path.

    Args:
        model_path: The path to the pre-trained huggingface model or local directory.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = AutoConfig.from_pretrained(model_path)
    model = Easyrec.from_pretrained(model_path, config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    return model, tokenizer


def compute_embeddings(
        sentences: List[str], 
        model: Easyrec, 
        tokenizer: AutoTokenizer, 
        batch_size: int = 8) -> torch.Tensor:
    """
    Compute embeddings for a list of sentences using the specified model and tokenizer.

    Args:
        sentences: A list of sentences for which to compute embeddings.
        model: The pre-trained model used for generating embeddings.
        tokenizer: The tokenizer used to preprocess the sentences.
        batch_size: The number of sentences to process in each batch (default is 8).

    Returns:
        torch.Tensor: A tensor containing the normalized embeddings for the input sentences.
    """

    embeddings = []
    count_sentences = len(sentences)
    device = next(model.parameters()).device  # Get the device on which the model is located

    for start in range(0, count_sentences, batch_size):
        end = start + batch_size
        batch_sentences = sentences[start:end]
        
        inputs = tokenizer(batch_sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()} # Move input tensors to the same device as the model
        
        with torch.inference_mode():
            outputs = model.encode(inputs['input_ids'], inputs['attention_mask'])
            batch_embeddings = F.normalize(outputs.pooler_output.detach().float(), dim=-1)
            
            embeddings.append(batch_embeddings.cpu())
    
    return torch.cat(embeddings, dim=0) # Concatenate all computed embeddings into a single tensor
