#!/usr/bin/env python3
import numpy as np
from sqlalchemy import text

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from mythologizer_postgres.store import (
    insert_mythemes_bulk,
    get_mythemes_bulk,
    get_mytheme,
)
from mythologizer_postgres.db import get_engine

def main():
    # Clean slate: truncate existing data
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE public.mythemes RESTART IDENTITY"))

    # Prepare test data (dim must match your VECTOR definition)
    dim = 384
    sentences = [f"theme_{i+1}" for i in range(3)]
    embeddings = np.arange(3 * dim, dtype=float).reshape(3, dim)

    # Insert test records
    print("Inserting test mythemes...")
    insert_mythemes_bulk(sentences, embeddings)

    # Fetch as Python lists
    print("\nFetch all themes (list mode):")
    ids, sents, embs = get_mythemes_bulk()
    print("IDs      :", ids)
    print("Sentences:", sents)
    print("Embeddings:", embs)

    # Fetch as NumPy array
    print("\nFetch all themes (NumPy mode):")
    ids_np, sents_np, embs_np = get_mythemes_bulk(as_numpy=True)
    print("IDs      :", ids_np)
    print("Sentences:", sents_np)
    print("Embeddings ndarray shape:", embs_np.shape)
    print(embs_np)

    # Fetch a single theme
    print("\nFetch single theme id=2:")
    id2, sent2, emb2 = get_mytheme(2, as_numpy=True)
    print("ID       :", id2)
    print("Sentence :", sent2)
    print("Embedding ndarray:", emb2)

if __name__ == "__main__":
    main()
