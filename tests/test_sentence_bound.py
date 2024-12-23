import logging
from nltk.tokenize import sent_tokenize
import asyncio

from openai import AsyncOpenAI

from chunking.validator.reward import check_chunk_ends_on_sentence_boundary
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

    for chunk in chunks:
        assert check_chunk_ends_on_sentence_boundary(document_sentences, chunk) == False

    assert check_chunk_ends_on_sentence_boundary(document_sentences, document) == True

    mid_chunks = mid_sentence_chunker(document, 100)

    for chunk in mid_chunks:
        assert check_chunk_ends_on_sentence_boundary(document_sentences, chunk) == False

    all_pageids = get_articles()

    documents = await get_or_load_synthetic_data(
        n=2,
        all_pageids=all_pageids,
        k=3,
        loop_range=range(3, 4),
        aclient=AsyncOpenAI(),
        synth_gen_batch_size=5,
    )

    for document, save_path in documents:
        document_sentences = sent_tokenize(document)
        logger.info(
            f"Testing document with {len(document)} characters from {save_path}"
        )
        mid_chunks = mid_sentence_chunker(document, 2048)
        logger.info(f"Created {len(mid_chunks)} mid-sentence chunks")
        for chunk in mid_chunks:
            assert (
                check_chunk_ends_on_sentence_boundary(document_sentences, chunk)
                == False
            )

        normal_chunks = base_chunker(document, 2048)
        logger.info(f"Created {len(normal_chunks)} base chunks")
        for chunk in normal_chunks:
            assert (
                check_chunk_ends_on_sentence_boundary(document_sentences, chunk) == True
            )


def test_sentence_bound():
    asyncio.run(main())
