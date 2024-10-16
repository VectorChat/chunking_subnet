import logging
from typing import List
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

def base_chunker(document: str, chunk_size: int) -> List[str]:
    logger.debug(f"chunking document of length {len(document)} with chunk size {chunk_size}")
    sentences = sent_tokenize(document)
    
    chunks = []       
    while len(sentences) > 0:
        chunks.append(sentences[0])
        del sentences[0]
        while len(sentences) > 0:
            if len(chunks[-1] + " " + sentences[0]) > chunk_size:
                # logger.debug(f"chunk of length {len(chunks[-1])} exceeds chunk size {chunk_size}, breaking")
                break
            # logger.debug(f"adding sentence to chunk of length {len(chunks[-1])}")
            chunks[-1] += (" " + sentences.pop(0))
    
    return chunks