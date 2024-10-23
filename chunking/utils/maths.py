import numpy as np

def calc_cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))