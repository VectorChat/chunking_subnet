import logging
import regex as re
from typing import List
from nltk.tokenize import sent_tokenize
from termcolor import colored

logger = logging.getLogger(__name__)


def base_chunker(document: str, chunk_size: int) -> List[str]:
    logger.debug(
        f"chunking document of length {len(document)} with chunk size {chunk_size}"
    )
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
            chunks[-1] += " " + sentences.pop(0)

    return chunks


def mid_sentence_chunker(document: str, chunk_size: int) -> List[str]:
    logger.debug(
        f"chunking document of length {len(document)} with chunk size {chunk_size}"
    )

    base_chunks = base_chunker(document, chunk_size)

    new_chunks = []

    for chunk in base_chunks:
        can_split_regex = r"([,;])"

        match = re.search(can_split_regex, chunk)
        if match:
            index = match.start()
            first_half = chunk[: index + 1]
            second_half = chunk[index + 1 :]
            logger.info(
                f"split chunk:\n\nold: {colored(chunk, 'yellow')}\n\nfirst_half: {colored(first_half, 'cyan')}\n\nsecond_half: {colored(second_half, 'green')}"
            )
            if len(first_half) > 0:
                new_chunks.append(first_half)
            if len(second_half) > 0:
                new_chunks.append(second_half)
        else:
            new_chunks.append(chunk)

    return new_chunks
