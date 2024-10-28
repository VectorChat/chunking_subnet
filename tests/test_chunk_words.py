import asyncio
import logging
from math import ceil
import time

from openai import AsyncOpenAI, OpenAI
from chunking.protocol import chunkSynapse
from chunking.validator.reward import (
    check_chunk_words_in_document,
    check_document_words_in_chunks,
    reward,
)
from chunking.validator.task_api import (
    generate_doc_normal,
    generate_synthetic_synapse,
    get_wiki_content_for_page,
)
from nltk.tokenize import sent_tokenize
import random

from tests.utils import get_articles, base_chunker
from tests.utils.test_cases import read_test_cases

logger = logging.getLogger(__name__)


def create_bad_chunk(chunk: str):
    # remove a random word from the chunk
    words = chunk.split()
    if len(words) == 1:
        return chunk
    index = random.randint(0, len(words) - 1)
    words.pop(index)
    return " ".join(words)


async def run_test():
    # courteousy of tvxq19910509
    test_document = (
        "Mammoths – Giants of the Ice Age (3 ed.). With some extra words here."
    )
    test_chunk = "Giants of the Ice Age (3 ed.)."

    logger.info(f"testing test_chunk: {test_chunk}")

    assert (
        check_chunk_words_in_document(test_chunk, test_document, verbose=True) == True
    )

    # test with static test cases

    chunk_size = 4096
    for case in read_test_cases():
        logger.info("testing case")
        document = case

        chunks = base_chunker(document, chunk_size)

        checks = [
            check_chunk_words_in_document(chunk, document, verbose=True)
            for chunk in chunks
        ]

        assert all(checks)

        assert check_document_words_in_chunks(document, chunks, chunk_size)

        sample_size = min(10, len(chunks))

        # create bad chunks by removing word from random chunk
        random_chunks = random.sample(chunks, sample_size)

        bad_chunks = [create_bad_chunk(chunk) for chunk in random_chunks]

        logger.info(f"created {len(bad_chunks)} bad chunks")

        checks = [
            check_chunk_words_in_document(chunk, document, verbose=True)
            for chunk in bad_chunks
        ]

        assert not any(checks)

        assert not check_document_words_in_chunks(document, bad_chunks, chunk_size)

    # test random articles

    articles = get_articles()

    sample_size = 10
    first_n = 2

    assert first_n < sample_size

    client = AsyncOpenAI()

    NUM_EMBEDDINGS = 2000

    random_articles = random.sample(articles, sample_size)

    additional_pageids = [26623321]

    articles_to_test = additional_pageids + random_articles

    logger.info(f"testing {len(articles_to_test)} articles")

    # for first n, also run with reward fn
    logger.info(f"testing {first_n} articles with reward fn")
    for i in range(first_n):
        logger.info(f"testing article {articles_to_test[i]}")
        document, title = await get_wiki_content_for_page(articles_to_test[i])

        chunks = base_chunker(document, chunk_size)

        checks = [
            check_chunk_words_in_document(chunk, document, verbose=True)
            for chunk in chunks
        ]

        assert all(checks)

        assert check_document_words_in_chunks(document, chunks, chunk_size)

        logger.info("passed chunk words check")

        chunk_qty = ceil(ceil(len(document) / chunk_size) * 1.5)

        synapse = chunkSynapse(
            document=document,
            chunk_size=chunk_size,
            chunk_qty=chunk_qty,
            chunks=chunks,
            time_soft_max=15.0,
        )

        logger.info("rewarding chunks")
        start_time = time.time()

        reward_value, _ = await reward(
            document=document,
            chunk_size=chunk_size,
            chunk_qty=chunk_qty,
            response=synapse,
            num_embeddings=NUM_EMBEDDINGS,
            client=client,
            verbose=True,
        )

        end_time = time.time()

        logger.info(f"rewarded chunks in {end_time - start_time} seconds")

        assert reward_value > 0

    for i, article in enumerate(articles_to_test):
        logger.info(f"testing article {i + 1}/{len(articles_to_test)}: {article}")
        document, title = await get_wiki_content_for_page(article)

        logger.info(f"title: {title}")

        chunks = base_chunker(document, 4096)

        checks = [
            check_chunk_words_in_document(chunk, document, verbose=True)
            for chunk in chunks
        ]

        assert all(checks)

        assert check_document_words_in_chunks(document, chunks, chunk_size)

    logger.info(f"finished checking {len(articles_to_test)} articles")


def test_chunk_words():
    asyncio.run(run_test())


# if __name__ == "__main__":
#     print("running tests")
#     test_chunk_words()
