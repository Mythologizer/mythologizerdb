import numpy as np
from typing import List, Optional, Tuple, Union, Sequence
from sqlalchemy import text
from sqlalchemy.engine import Engine

from mythologizer_postgres.db import get_engine

EngineT = Engine


def get_mythemes_bulk(
    ids: Optional[List[int]] = None,
    as_numpy: bool = True,
) -> Tuple[List[int], List[str], Union[List[List[float]], np.ndarray]]:
    """
    Fetch mythemes by id or all of them.
    Returns three parallel lists: ids, sentences, embeddings.
    If as_numpy is True, embeddings come back as a single numpy.ndarray
    of shape (n_mythemes, embedding_dim).
    """
    engine: EngineT = get_engine()

    if ids:
        placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
        sql_text = text(f"""
            SELECT id, sentence, embedding
              FROM public.mythemes
             WHERE id IN ({placeholders})
        """)
        bind = {f"id_{i}": val for i, val in enumerate(ids)}
        with engine.connect() as conn:
            rows = conn.execute(sql_text, bind).all()
    else:
        sql_text = text("SELECT id, sentence, embedding FROM public.mythemes")
        with engine.connect() as conn:
            rows = conn.execute(sql_text).all()

    if not rows:
        if as_numpy:
            return [], [], np.empty((0, 0))
        return [], [], []

    ids_list, sentences_list, raw_embs = zip(*rows)
    embs: List[List[float]] = [list(map(float, emb)) for emb in raw_embs]

    if as_numpy:
        return list(ids_list), list(sentences_list), np.array(embs, dtype=float)

    return list(ids_list), list(sentences_list), embs


def get_mytheme(
    theme_id: int,
    as_numpy: bool = True,
) -> Tuple[int, str, Union[List[float], np.ndarray]]:
    """
    Fetch exactly one mytheme by id.
    Raises KeyError if not found.
    """
    ids, sentences, embs = get_mythemes_bulk([theme_id], as_numpy=as_numpy)
    if not ids:
        raise KeyError(f"mytheme {theme_id} not found")
    return ids[0], sentences[0], embs[0]


def insert_mythemes_bulk(
    sentences: Sequence[str],
    embeddings: Union[Sequence[Sequence[float]], np.ndarray],
) -> None:
    """
    Insert multiple mythemes in one transaction.
    sentences[i] pairs with embeddings[i].
    Accepts embeddings as a numpy.ndarray (n, dim) or list of lists.
    """
    engine: EngineT = get_engine()

    if isinstance(embeddings, np.ndarray):
        embeddings_list = embeddings.tolist()
    else:
        embeddings_list = embeddings

    sql_text = text("""
        INSERT INTO public.mythemes (sentence, embedding)
        VALUES (:sentence, :embedding)
    """)
    records = [
        {"sentence": s, "embedding": e}
        for s, e in zip(sentences, embeddings_list)
    ]

    with engine.begin() as conn:
        conn.execute(sql_text, records)
