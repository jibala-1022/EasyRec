#!/bin/bash

models=(
    "hkuds/easyrec-roberta-small"
    "hkuds/easyrec-roberta-base"
    "hkuds/easyrec-roberta-large"
    "FacebookAI/roberta-base"
    "FacebookAI/roberta-large"
    "google-bert/bert-base-uncased"
    "google-bert/bert-large-uncased"
    "princeton-nlp/sup-simcse-roberta-base"
    "princeton-nlp/sup-simcse-roberta-large"
)

for model in "${models[@]}"
do
    python encode_easyrec.py --model "$model" --cuda 0
done
