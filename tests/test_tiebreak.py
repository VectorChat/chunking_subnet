import asyncio
from copy import deepcopy
from math import ceil
import random
from typing import List
import numpy as np
import bittensor as bt
from openai import OpenAI

from chunking.protocol import chunkSynapse
from chunking.utils.chunks import calculate_chunk_qty
from chunking.validator.reward import get_chunks_hash, get_rewards
from chunking.validator.task_api import get_wiki_content_for_page
from tests.utils.articles import get_articles
from tests.utils.chunker import base_chunker


async def run_test():
    bt.debug()

    articles = get_articles()

    test_pageid = random.choice(articles)

    test_doc, title = await get_wiki_content_for_page(test_pageid)

    chunk_size = 4096
    chunk_qty = calculate_chunk_qty(test_doc, chunk_size)

    base_synapse = chunkSynapse(
        document=test_doc, chunk_size=chunk_size, chunk_qty=chunk_qty, time_soft_max=15
    )

    chunk_methods = [
        lambda: base_chunker(test_doc, chunk_size),
        lambda: base_chunker(test_doc, chunk_size + 100),
        lambda: base_chunker(test_doc, chunk_size - 400),
        lambda: base_chunker(test_doc, chunk_size - 800),
        lambda: base_chunker(test_doc, chunk_size - 1600),
    ]

    def run_chunk_method(i: int):
        assert len(chunk_methods) > i

        chunk_method = chunk_methods[i]
        return chunk_method()

    # all different

    chunk_results = [run_chunk_method(i) for i in range(2)]

    assert all(len(chunk_result) > 0 for chunk_result in chunk_results)

    chunk_hashes = [get_chunks_hash(chunk_result) for chunk_result in chunk_results]

    assert len(set(chunk_hashes)) == len(chunk_hashes)

    def get_chunk_responses(chunk_results: List[List[str]]) -> List[chunkSynapse]:
        chunk_responses: list[chunkSynapse] = []

        for chunk_result in chunk_results:
            synapse = deepcopy(base_synapse)
            synapse.chunks = chunk_result
            chunk_responses.append(synapse)

        return chunk_responses

    chunk_responses = get_chunk_responses(chunk_results)

    assert all(
        chunk_response is not None
        and chunk_response.chunks is not None
        and len(chunk_response.chunks) > 0
        for chunk_response in chunk_responses
    )

    override_client = OpenAI()
    override_num_embeddings = 50

    rewards, extra_infos = get_rewards(
        None,
        test_doc,
        chunk_size,
        chunk_qty,
        chunk_responses,
        override_client,
        override_num_embeddings,
    )

    assert len(set(rewards.tolist())) == len(chunk_results)

    # two people tie

    chunk_results = []

    chunk_results.append(run_chunk_method(0))
    chunk_results.append(run_chunk_method(1))
    chunk_results.append(run_chunk_method(1))
    chunk_results.append(run_chunk_method(3))

    chunk_responses = get_chunk_responses(chunk_results)

    rewards, extra_infos = get_rewards(
        None,
        test_doc,
        chunk_size,
        chunk_qty,
        chunk_responses,
        override_client,
        override_num_embeddings,
    )

    assert len(set(rewards.tolist())) == 3

    assert rewards[1] == rewards[2]
    assert rewards[1] != rewards[0]
    assert rewards[1] != rewards[3]

    # three people tie

    chunk_results = []

    chunk_results.append(run_chunk_method(0))
    chunk_results.append(run_chunk_method(1))
    chunk_results.append(run_chunk_method(1))
    chunk_results.append(run_chunk_method(3))

    chunk_responses = get_chunk_responses(chunk_results)

    rewards, extra_infos = get_rewards(
        None,
        test_doc,
        chunk_size,
        chunk_qty,
        chunk_responses,
        override_client,
        override_num_embeddings,
    )

    assert len(set(rewards.tolist())) == 3

    assert rewards[1] == rewards[2]
    assert rewards[1] != rewards[0]
    assert rewards[1] != rewards[3]

    # 2 sep groups tie

    chunk_results = []

    chunk_results.append(run_chunk_method(0))
    chunk_results.append(run_chunk_method(1))
    chunk_results.append(run_chunk_method(1))
    chunk_results.append(run_chunk_method(2))
    chunk_results.append(run_chunk_method(2))
    chunk_results.append(run_chunk_method(3))

    chunk_responses = get_chunk_responses(chunk_results)

    rewards, extra_infos = get_rewards(
        None,
        test_doc,
        chunk_size,
        chunk_qty,
        chunk_responses,
        override_client,
        override_num_embeddings,
    )

    assert len(set(rewards.tolist())) == 4

    assert rewards[1] == rewards[2]
    assert rewards[3] == rewards[4]
    assert rewards[1] != rewards[0]
    assert rewards[3] != rewards[0]
    assert rewards[1] != rewards[3]


def test_tiebreak():
    asyncio.run(run_test())
