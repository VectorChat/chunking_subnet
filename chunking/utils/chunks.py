from math import ceil

def calculate_chunk_qty(document: str, chunk_size: int) -> int:
    return ceil(ceil(len(document) / chunk_size) * 1.5)