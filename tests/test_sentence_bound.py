import logging
from nltk.tokenize import sent_tokenize
import asyncio

from openai import AsyncOpenAI

from chunking.validator.reward import check_chunks_end_on_sentence_boundaries
from tests.utils.articles import get_articles
from tests.utils.chunker import base_chunker, mid_sentence_chunker
from tests.utils.synthetic import get_or_load_synthetic_data
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


async def main():
    document = "Yet, like these creatures, M. rotula is an integral part of its environment, contributing to the decomposition of organic matter and the cycling of nutrients within forest ecosystems. "

    chunks = [
        "Yet, like these creatures, M.",
        "rotula is an integral part of its environment, contributing to the decomposition of organic matter and the cycling of nutrients within forest ecosystems. ",
    ]
    document_sentences = sent_tokenize(document)

    assert check_chunks_end_on_sentence_boundaries(chunks, document_sentences) == False

    mid_chunks = mid_sentence_chunker(document, 100)

    assert check_chunks_end_on_sentence_boundaries(mid_chunks, document_sentences) == False

    all_pageids = get_articles()

    documents = await get_or_load_synthetic_data(
        n=20,
        all_pageids=all_pageids,
        k=3,
        loop_range=range(3, 4),
        aclient=AsyncOpenAI(),
        synth_gen_batch_size=5,
    )

    test_case_docs = []
    prefix = "tests/test_cases/sentence_bound_"
    for i in range(1, 6):
        path = f"{prefix}{i}.txt"
        with open(path, "r") as file:
            document = file.read()
            test_case_docs.append((document, path))

    all_docs = test_case_docs + documents

    for i, (document, save_path) in enumerate(all_docs):
        document_sentences = sent_tokenize(document)
        logger.info(
            f"Testing document with {len(document)} characters from {save_path}"
        )
        mid_chunks = mid_sentence_chunker(document, 2048)
        logger.info(f"Created {len(mid_chunks)} mid-sentence chunks")

        assert check_chunks_end_on_sentence_boundaries(mid_chunks, document_sentences) == False, f"Mid sentence chunks do end on sentence boundaries for document {i}"
        logger.info("Mid sentence chunks do not end on sentence boundaries")

        normal_chunks = base_chunker(document, 2048)
        logger.info(f"Created {len(normal_chunks)} base chunks")
        assert check_chunks_end_on_sentence_boundaries(normal_chunks, document_sentences) == True, f"Base chunks do not end on sentence boundaries for document {i}"
        logger.info("Base chunks end on sentence boundaries")

    logger.info(f"Passed {i} documents")

def test_sentence_bound():
    asyncio.run(main())
