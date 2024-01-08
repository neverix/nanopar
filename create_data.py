from sentencepiece import SentencePieceProcessor
from streaming import MDSWriter
import pandas as pd
import numpy as np

from tqdm.auto import trange
import fire

from pathlib import Path
import random
import gzip
import json
import os


def main(
    model_dir = Path("llama-2-7b"),
    train_ds_size = 1024,
    seq_len = 2048,
    use_pile=False,
    data_dir=Path("data")):
    tokenizer = SentencePieceProcessor(str(model_dir / "tokenizer.model"))

    if use_pile:
        pile = pd.read_parquet("pile.parquet")
        tokens = []
        for i in trange(min(train_ds_size, len(pile))):
            tokens.extend(tokenizer.Encode(pile.iloc[i, 0]))
        tokens = np.asarray(tokens, dtype=np.int32)
        with MDSWriter(out=data_dir, columns={"tokens": "ndarray"}, compression="zstd") as out:
            for _ in trange(train_ds_size):
                offset = random.randrange(0, len(tokens) - seq_len)
                sample = {
                    "tokens": tokens[offset:offset+seq_len]
                }
                out.write(sample)
    else:
        hh = gzip.open("hh-rlhf/helpful-base/train.jsonl.gz", mode="rt")
        samples = []
        
        def encode(text):
            tokens = tokenizer.Encode(text)[-seq_len:]
            tokens = np.asarray(tokens, dtype=np.int32)
            tokens = np.pad(tokens, ((0, seq_len - len(tokens)),), constant_values=-100)
            return tokens
        
        for i, sample in zip(trange(train_ds_size), hh):
            sample = json.loads(sample.strip())
            samples.append((encode(sample["chosen"]), encode(sample["rejected"])))
        if not os.path.exists(data_dir):
            with MDSWriter(out=str(data_dir), columns={"tokens": "ndarray"}, compression="zstd") as out:
                for chosen, rejected in samples:
                    out.write({
                        "tokens": np.stack((chosen, rejected), axis=0)
                    })


if __name__ == "__main__":
    fire.Fire(main)
