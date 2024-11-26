from math import ceil

def calculate_chunk_qty(document: str, chunk_size: int, multiplier: float = 2.0) -> int:
    return ceil(ceil(len(document) / chunk_size) * multiplier)
