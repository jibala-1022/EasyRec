import torch
from model import Easyrec
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL = "jibala-1022/my-easyrec-base"
config = AutoConfig.from_pretrained(MODEL)
model = Easyrec.from_pretrained(MODEL, config=config,)
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False,)

profiles = [
    'This user is a basketball fan and likes to play basketball and watch NBA games.', # user
    'This basketball draws in NBA enthusiasts.', # item 1
    'This item is nice for swimming lovers.'     # item 2
]

inputs = tokenizer(profiles, padding=True, truncation=True, max_length=512, return_tensors="pt")
with torch.inference_mode():
    embeddings = model.encode(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
embeddings = F.normalize(embeddings.pooler_output.detach().float(), dim=-1)

print(embeddings[0] @ embeddings[1])    # 0.8576
print(embeddings[0] @ embeddings[2])    # 0.2171