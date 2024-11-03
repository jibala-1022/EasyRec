#!/bin/bash

models=(
    "easyrec-roberta-small"
    "easyrec-roberta-base"
    "easyrec-roberta-large"
    "roberta-base"
    "roberta-large"
    "bert-base-uncased"
    "bert-large-uncased"
    "sup-simcse-roberta-base"
    "sup-simcse-roberta-large"
)

for cf_model in "gccf" "lightgcn"
do
    for lang_model in "${models[@]}"
    do
        python run.py --model "$cf_model"_plus --semantic "$lang_model" --dataset movies --cuda 0
    done
done
