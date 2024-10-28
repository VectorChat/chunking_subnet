import asyncio
import logging
from math import ceil
from random import sample
from openai import AsyncOpenAI, OpenAI
from chunking.protocol import chunkSynapse
from chunking.validator.reward import reward
from chunking.validator.task_api import generate_doc_normal, generate_synthetic_synapse

from tests.utils import base_chunker

logger = logging.getLogger(__name__)

async def run_test():
    PAGE_ID = 33653136

    tuple = await generate_doc_normal(None, PAGE_ID)

    document = tuple[0]

    timeout = 20
    time_soft_max = timeout * 0.75
    chunk_size = 4096
    chunk_qty = ceil(ceil(len(document) / chunk_size) * 1.5)
    synapse = chunkSynapse(
        document=document,
        time_soft_max=time_soft_max,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        timeout=timeout,
    )

    assert synapse.time_soft_max == 15

    client = AsyncOpenAI()

    NUM_EMBEDDINGS = 150

    # reward should be 0 if no chunks
    logger.info("reward should be 0 if no chunks")

    reward_value, _ = await reward(
        document=document,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        response=synapse,
        num_embeddings=NUM_EMBEDDINGS,
        client=client,
        verbose=True,
    )

    assert reward_value == 0


    # reward should be zero if any word is reordered

    logger.info("reward should be zero if any word is reordered")
    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    ## shuffle the words in the first chunk

    test_chunks[0] = " ".join(
        sample(test_chunks[0].split(), len(test_chunks[0].split()))
    )

    synapse.chunks = test_chunks

    reward_value, _ = await reward(
        document=document,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        response=synapse,
        num_embeddings=NUM_EMBEDDINGS,
        client=client,
        verbose=True,
    )

    assert reward_value == 0

    # reward should be zero if any word is removed

    logger.info("reward should be zero if any word is removed")

    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    ## remove the first word from the first chunk

    test_chunks[0] = " ".join(test_chunks[0].split()[1:])

    synapse.chunks = test_chunks

    reward_value, _ = await reward(
        document=document,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        response=synapse,
        num_embeddings=NUM_EMBEDDINGS,
        client=client,
        verbose=True,
    )

    assert reward_value == 0

    # reward should be zero if any word is added

    logger.info("reward should be zero if any word is added")

    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    ## add a word to the first chunk

    test_chunks[0] = " ".join(test_chunks[0].split() + ["word"])

    synapse.chunks = test_chunks

    reward_value, _ = await reward(
        document=document,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        response=synapse,
        num_embeddings=NUM_EMBEDDINGS,
        client=client,
        verbose=True,
    )

    assert reward_value == 0

    # reward should be zero if any chunks are removed

    logger.info("reward should be zero if any chunks are removed")

    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    ## remove the first chunk

    test_chunks = test_chunks[1:]

    synapse.chunks = test_chunks

    reward_value, _ = await reward(
        document=document,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        response=synapse,
        num_embeddings=NUM_EMBEDDINGS,
        client=client,
        verbose=True,
    )

    assert reward_value == 0

    # should give reward for proper chunking

    logger.info("reward should be non-zero for proper chunking")

    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    synapse.chunks = test_chunks

    reward_value, _ = await reward(
        document=document,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        response=synapse,
        num_embeddings=NUM_EMBEDDINGS,
        client=client,
        verbose=True,
    )

    assert reward_value > 0


def test_reward_fn():
    asyncio.run(run_test())
