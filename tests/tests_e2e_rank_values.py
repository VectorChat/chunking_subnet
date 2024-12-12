import asyncio
from copy import deepcopy
import logging
import random
from typing import List
import numpy as np
from openai import AsyncOpenAI, OpenAI

from chunking.protocol import chunkSynapse
from chunking.utils.chunks import calculate_chunk_qty
from chunking.utils.score import get_rank_value_to_count
from chunking.utils.synthetic.synthetic import get_wiki_content_for_page
from chunking.validator.reward import (
    get_chunks_hash,
    get_rewards,
    rank_responses,
    rank_responses_global,
)
from chunking.validator.tournament import create_groups, get_alpha
from tests.utils.articles import get_articles
from tests.utils.chunker import base_chunker
from tests.utils.misc import compare_lists
import bittensor as bt

ALPHA = 0.025
GROUP_SIZE = 2

logger = logging.getLogger(__name__)


async def main():
    bt.logging.set_debug()
    rankings = np.arange(256)
    scores = rankings / 2

    # make miner groups and rank values
    miner_groups, group_ranks, group_rank_values = create_groups(rankings, GROUP_SIZE)

    miner_group_index = 1

    # get fake responses for a miner group
    articles = get_articles()

    test_pageid = random.choice(articles)

    test_doc, title = await get_wiki_content_for_page(test_pageid)

    logger.info(f"Title: {title}")

    chunk_size = 4096
    chunk_qty = calculate_chunk_qty(test_doc, chunk_size)

    base_synapse = chunkSynapse(
        document=test_doc, chunk_size=chunk_size, chunk_qty=chunk_qty, time_soft_max=15
    )

    chunk_methods = [
        lambda: base_chunker(test_doc, chunk_size),
        lambda: base_chunker(test_doc, chunk_size - 400),
        lambda: base_chunker(test_doc, chunk_size - 800),
        lambda: base_chunker(test_doc, chunk_size - 1600),
    ]

    def run_chunk_method(i: int):
        assert len(chunk_methods) > i

        chunk_method = chunk_methods[i]
        return chunk_method()

    chunk_results = []

    chunk_results.append(run_chunk_method(0))
    chunk_results.append(run_chunk_method(1))
    chunk_results.append(run_chunk_method(1))
    chunk_results.append(run_chunk_method(2))

    target_unique = 3

    assert all(len(chunk_result) > 0 for chunk_result in chunk_results)

    chunk_hashes = [get_chunks_hash(chunk_result) for chunk_result in chunk_results]

    assert len(set(chunk_hashes)) == target_unique

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

    # get rewards (make sure tieing system works)
    override_client = AsyncOpenAI()
    override_num_embeddings = 50

    rewards, extra_infos = await get_rewards(
        document=test_doc,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        responses=chunk_responses,
        num_embeddings=override_num_embeddings,
        client=override_client,
    )

    logger.info(f"rewards: {rewards}")
    assert len(set(rewards.tolist())) == target_unique

    # get local group ranks
    local_group_ranks = rank_responses(rewards)

    logger.info(f"local_group_ranks: {local_group_ranks}")

    assert len(set(local_group_ranks.tolist())) == target_unique

    # get global rank values (make sure they are correct)
    rank_values_for_group = group_rank_values[miner_group_index]
    assert compare_lists(rank_values_for_group, [0.5, 1.5, 2.5, 3.5])

    logger.info(f"rank_values_for_group: {rank_values_for_group}")

    global_rank_values = rank_responses_global(
        None,
        rank_values_for_group,
        local_group_ranks,
        miner_groups[miner_group_index],
        scores,
    )

    logger.info(f"global_rank_values: {global_rank_values}")

    assert len(set(global_rank_values.tolist())) == target_unique

    rank_value_to_count = get_rank_value_to_count(global_rank_values)

    logger.info(f"rank_value_to_count: {get_rank_value_to_count(global_rank_values)}")

    assert (
        len(set(rank_value_to_count.values()))
        == len(chunk_results) - target_unique + 1
    )


def test_e2e_rank_values():
    asyncio.run(main())
