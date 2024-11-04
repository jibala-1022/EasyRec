#!/bin/bash

models=(
    # HF models
    "easyrec-roberta-large"
    "roberta-base"
    "roberta-large"
    "bert-base-uncased"
    "bert-large-uncased"
    "sup-simcse-roberta-base"
    "sup-simcse-roberta-large"
    # Local models
    "easyrec-small"
    "easyrec-base"
)

for model_name in "${models[@]}"
do
    python eval_text_emb.py --model "$model_name" --cuda 0
done
