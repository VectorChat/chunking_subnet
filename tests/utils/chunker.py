from typing import List
from nltk.tokenize import sent_tokenize


def base_chunker(document: str, chunk_size: int) -> List[str]:
    sentences = sent_tokenize(document)
    
    chunks = []       
    while len(sentences) > 0:
        chunks.append(sentences[0])
        del sentences[0]
        while len(sentences) > 0:
            if len(chunks[-1] + " " + sentences[0]) > chunk_size:
                break
            chunks[-1] += (" " + sentences.pop(0))
    
    return chunks