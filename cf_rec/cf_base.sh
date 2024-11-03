#!/bin/bash

for cf_model in "gccf" "lightgcn"
do
    python run.py --model "$cf_model" --dataset movies --cuda 0
done
