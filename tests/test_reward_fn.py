import asyncio
import logging
from math import ceil, e
from random import sample
from openai import AsyncOpenAI, OpenAI
import pytest
from chunking.protocol import chunkSynapse
from chunking.utils.chunks import calculate_chunk_qty
from chunking.utils.synthetic.synthetic import generate_doc_normal
from chunking.validator.reward import reward

from tests.utils.articles import get_articles
from tests.utils.chunker import base_chunker, mid_sentence_chunker
from tests.utils.synthetic import get_or_load_synthetic_data
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
import bittensor as bt

load_dotenv()

logger = logging.getLogger(__name__)


async def run_test(document: str):
    bt.debug()
    bt.logging.debug(f"running test with document of length {len(document)}")

    timeout = 20
    time_soft_max = timeout * 0.75
    chunk_size = 4096
    chunk_qty = calculate_chunk_qty(
        document=document,
        chunk_size=chunk_size,
    )
    synapse = chunkSynapse(
        document=document,
        time_soft_max=time_soft_max,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        timeout=timeout,
    )

    client = AsyncOpenAI()

    NUM_EMBEDDINGS = 150

    async def _calculate_reward(
        synapse: chunkSynapse, do_checks: bool, do_penalties: bool
    ):
        reward_value, extra_info = await reward(
            document=document,
            chunk_size=chunk_size,
            chunk_qty=chunk_qty,
            response=synapse,
            num_embeddings=NUM_EMBEDDINGS,
            client=client,
            verbose=True,
            do_checks=do_checks,
            do_penalties=do_penalties,
        )

        return reward_value, extra_info

    assert synapse.time_soft_max == 15

    # reward should be 0 if no chunks
    logger.info("reward should be 0 if no chunks")

    reward_value, _ = await _calculate_reward(
        synapse, do_checks=True, do_penalties=False
    )

    assert reward_value == 0

    logger.info("reward should be zero if chunk does not split on sentence boundary")

    synapse.chunks = mid_sentence_chunker(synapse.document, synapse.chunk_size)

    logger.info(f"got {len(synapse.chunks)} chunks")

    reward_value, _ = await _calculate_reward(
        synapse, do_checks=True, do_penalties=True
    )

    assert reward_value == 0

    logger.info(
        "reward should be non-zero if not doing checks and chunk does not end on sentence boundary"
    )

    reward_value, _ = await _calculate_reward(
        synapse, do_checks=False, do_penalties=True
    )

    assert reward_value > 0

    # reward should be zero if any word is reordered

    logger.info("reward should be zero if any word is reordered")
    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    ## shuffle the words in the first chunk

    test_chunks[0] = " ".join(
        sample(test_chunks[0].split(), len(test_chunks[0].split()))
    )

    synapse.chunks = test_chunks

    reward_value, _ = await _calculate_reward(
        synapse, do_checks=True, do_penalties=False
    )

    assert reward_value == 0

    logger.info("reward should be non-zero if not doing checks")

    reward_value, _ = await _calculate_reward(
        synapse, do_checks=False, do_penalties=False
    )

    assert reward_value > 0

    # reward should be zero if any word is removed

    logger.info("reward should be zero if any word is removed")

    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    ## remove the words from the first chunk
    first_chunk = test_chunks[0]
    logger.info(f"first_chunk: {first_chunk}")
    words = word_tokenize(first_chunk)
    words = words[2:]
    test_chunks[0] = " ".join(words)
    logger.info(f"new chunk after removing words: {test_chunks[0]}")

    synapse.chunks = test_chunks

    reward_value, _ = await _calculate_reward(
        synapse, do_checks=True, do_penalties=False
    )

    assert reward_value == 0

    logger.info("reward should be non-zero if not doing checks")

    reward_value, _ = await _calculate_reward(
        synapse, do_checks=False, do_penalties=False
    )

    assert reward_value > 0

    # reward should be zero if any word is added

    logger.info("reward should be zero if any word is added")

    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    ## add a word to the first chunk

    test_chunks[0] = " ".join(test_chunks[0].split() + ["word"])

    synapse.chunks = test_chunks

    reward_value, _ = await _calculate_reward(
        synapse, do_checks=True, do_penalties=False
    )

    assert reward_value == 0

    logger.info("reward should be non-zero if not doing checks")

    reward_value, _ = await _calculate_reward(
        synapse, do_checks=False, do_penalties=False
    )

    assert reward_value > 0

    # reward should be zero if any chunks are removed

    logger.info("reward should be zero if any chunks are removed")

    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    ## remove the first chunk

    test_chunks = test_chunks[1:]

    synapse.chunks = test_chunks

    reward_value, _ = await _calculate_reward(
        synapse, do_checks=True, do_penalties=False
    )

    assert reward_value == 0

    reward_value, _ = await _calculate_reward(
        synapse, do_checks=False, do_penalties=False
    )

    assert reward_value > 0

    # should give reward for proper chunking

    logger.info("reward should be non-zero for proper chunking")

    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    logger.info(f"got {len(test_chunks)} chunks")

    synapse.chunks = test_chunks

    reward_value, extra_info = await _calculate_reward(
        synapse, do_checks=True, do_penalties=False,
    )

    logger.info(f"num_embed_tokens: {extra_info['num_embed_tokens']}")
    embedding_reward = extra_info["embedding_reward"]
    logger.info(f"embedding_reward: {embedding_reward}")

    qty_penalty = extra_info["qty_penalty"]
    logger.info(f"qty_penalty: {qty_penalty}")

    size_penalty = extra_info["size_penalty"]
    logger.info(f"size_penalty: {size_penalty}")

    assert reward_value > 0
    assert size_penalty == 0
    assert qty_penalty == 0

    # should penalize big_chunks

    logger.info("should penalize big_chunks")

    big_chunks = base_chunker(synapse.document, synapse.chunk_size * 2)

    logger.info(f"got {len(big_chunks)} chunks")

    synapse.chunks = big_chunks

    reward_value, extra_info = await _calculate_reward(
        synapse, do_checks=True, do_penalties=True
    )

    embedding_reward = extra_info["embedding_reward"]
    logger.info(f"embedding_reward: {embedding_reward}")

    size_penalty = extra_info["size_penalty"]
    logger.info(f"size_penalty: {size_penalty}")

    assert size_penalty > 0

    logger.info("reward should not penalize big chunks if not doing penalties")

    reward_value, extra_info = await _calculate_reward(
        synapse, do_checks=True, do_penalties=False
    )

    size_penalty = extra_info["size_penalty"]
    logger.info(f"size_penalty: {size_penalty}")

    assert size_penalty == 0
    assert reward_value > 0

    # should penalize lots of chunks

    logger.info("should penalize lots of chunks")

    small_chunks = base_chunker(synapse.document, synapse.chunk_size // 4)

    logger.info(f"got {len(small_chunks)} chunks")

    synapse.chunks = small_chunks

    reward_value, extra_info = await _calculate_reward(
        synapse, do_checks=True, do_penalties=True
    )

    embedding_reward = extra_info["embedding_reward"]
    qty_penalty = extra_info["qty_penalty"]

    logger.info(f"embedding_reward: {embedding_reward}")
    logger.info(f"qty_penalty: {qty_penalty}")

    assert reward_value > 0
    assert reward_value < 1.1
    assert qty_penalty > 0

    logger.info("reward should not penalize lots of chunks if not doing penalties")

    logger.info(f"got {len(small_chunks)} chunks")

    reward_value, extra_info = await _calculate_reward(
        synapse, do_checks=True, do_penalties=False
    )

    qty_penalty = extra_info["qty_penalty"]

    assert qty_penalty == 0
    assert reward_value > 0


async def main(num_articles: int):
    # PAGE_ID = 33653136
    # tuple = await generate_doc_normal(None, PAGE_ID)
    # document = tuple[0]

    all_pageids = get_articles()

    logger.info(f"Testing reward function with {num_articles} articles")

    documents = await get_or_load_synthetic_data(
        n=num_articles,
        all_pageids=all_pageids,
        k=3,
        loop_range=range(3, 4),
        aclient=AsyncOpenAI(),
        synth_gen_batch_size=5,
    )

    logger.info(f"Got {len(documents)} documents")

    batch_size = 5

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]

        for document, save_path in batch:
            logger.info(
                f"Testing document with {len(document)} characters from {save_path}"
            )
            await run_test(document)
            logger.info(
                f"Finished testing document with {len(document)} characters from {save_path}"
            )


# @pytest.mark.parametrize("num_articles", [5])
def test_reward_fn(num_articles: int):
    asyncio.run(main(num_articles))
