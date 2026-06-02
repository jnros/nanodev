"""
Prepare enwik8 for character-level LM.
Split: first 90M chars train, next 5M val, last 5M test.
Only train.bin and val.bin are written (test set held out).
Same tokenization style as shakespeare_char (char→int, no BPE).
"""
import os
import pickle
import zipfile
import requests
import numpy as np

DATA_URL = 'http://mattmahoney.net/dc/enwik8.zip'
DIR = os.path.dirname(__file__)
ZIP_PATH = os.path.join(DIR, 'enwik8.zip')
RAW_PATH = os.path.join(DIR, 'enwik8')

if not os.path.exists(RAW_PATH):
    if not os.path.exists(ZIP_PATH):
        print("downloading enwik8.zip (~36 MB)...")
        r = requests.get(DATA_URL, stream=True)
        r.raise_for_status()
        with open(ZIP_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        print("download complete")
    print("extracting...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extract('enwik8', DIR)
    print("extracted")

with open(RAW_PATH, 'r', encoding='utf-8', errors='replace') as f:
    data = f.read()

n = len(data)
print(f"total chars: {n:,}")
assert n >= 90_000_000, f"expected >=90M chars, got {n:,}"

# proportional 90/5/5 split (enwik8 is 100M bytes; char count differs slightly)
t1 = int(n * 0.90)
t2 = int(n * 0.95)
train_data = data[:t1]
val_data   = data[t1:t2]
# test_data = data[t2:]  # held out

chars = sorted(set(data))
vocab_size = len(chars)
print(f"vocab size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids   = np.array(encode(val_data),   dtype=np.uint16)
print(f"train tokens: {len(train_ids):,}")
print(f"val tokens:   {len(val_ids):,}")

train_ids.tofile(os.path.join(DIR, 'train.bin'))
val_ids.tofile(os.path.join(DIR, 'val.bin'))

meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
with open(os.path.join(DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("saved train.bin, val.bin, meta.pkl")
