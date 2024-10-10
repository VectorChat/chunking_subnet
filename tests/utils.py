from nltk.tokenize import sent_tokenize

def base_chunker(text: str, chunk_size: int):
    document = sent_tokenize(text)
    chunks = []
    while len(document) > 0:
        chunks.append(document[0])
        del document[0]
        while len(document) > 0:
            if len(chunks[-1] + " " + document[0]) > chunk_size:
                break
            chunks[-1] += " " + document.pop(0)
    return chunks

