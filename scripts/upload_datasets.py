import torch
import os
from transformers import AutoTokenizer
from tqdm import tqdm
import json


# Upload monology/pile uncopyrighted from local
def upload_monology_pile():
    filepath = "gemmascope_tokens.pt"  # 36864 x 128
    BATCH_SIZE = 36864

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    dataset = torch.load(filepath, map_location="cpu")

    os.makedirs("monology_pile", exist_ok=True)

    texts = []
    for s in tqdm(dataset):
        texts.append([])
        for c in s:
            texts[-1].append(tokenizer.decode(c))
    with open("monology_pile/text.json", "w") as f:
        json.dump(texts, f)
    torch.save(dataset.clone(), "monology_pile/tokens.pt")
    os.system(f"modal volume put rqae-volume monology_pile/ datasets/monology_pile/")


if __name__ == "__main__":
    upload_monology_pile()
