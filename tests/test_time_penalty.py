import copy
import logging
import random
import time
import bittensor as bt
import asyncio

from openai import AsyncOpenAI

from chunking.protocol import chunkSynapse
from chunking.utils.chunks import calculate_chunk_qty
from chunking.utils.integrated_api.chunk.types import RewardOptions
from chunking.utils.synthetic.synthetic import get_wiki_content_for_page
from chunking.validator.reward import get_rewards
from tests.utils.articles import get_articles
from bittensor.core.settings import version_as_int

from tests.utils.chunker import base_chunker

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()


async def main():

    # bt.debug()

    articles = get_articles()

    test_pageid = random.choice(articles)

    test_doc, title = await get_wiki_content_for_page(test_pageid)

    logger.info(
        f"Got doc '{title}' (pageid: {test_pageid}) with {len(test_doc)} characters: {test_doc[:100]}..."
    )

    chunk_size = 3000
    chunk_qty = calculate_chunk_qty(test_doc, chunk_size)

    vali_wallet = bt.wallet(name="owner-localnet", hotkey="validator1")

    vali_dendrite = bt.dendrite(wallet=vali_wallet)

    synapse = chunkSynapse(
        document=test_doc, chunk_size=chunk_size, chunk_qty=chunk_qty, time_soft_max=15
    )
    synapse.dendrite = bt.TerminalInfo(
        ip=vali_dendrite.external_ip,
        version=version_as_int,
        nonce=time.time_ns(),
        uuid=vali_dendrite.uuid,
        hotkey=vali_dendrite.keypair.ss58_address,
    )

    # Build the Axon headers using the target axon's details
    synapse.axon = bt.TerminalInfo(
        ip="127.0.0.1",
        port=3000,
        hotkey=vali_dendrite.keypair.ss58_address,
    )

    synapse.time_soft_max = 15

    synapse.chunks = base_chunker(test_doc, chunk_size)



    hotkey_1 = "5GKH9FPPnWSUoeeTJp19wVtd84XqFW4pyK2ijV2GsFbhTrP1"
    hotkey_2 = "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3" 

    responses = [copy.deepcopy(synapse) for _ in range(2)]

    responses[0].axon.hotkey = hotkey_1
    responses[1].axon.hotkey = hotkey_2

    logger.info("Identical responses with different process times where one exceeds the time soft max should have different final rewards")

    responses[0].dendrite.process_time = 10
    responses[1].dendrite.process_time = 20

    rewards, extra_infos = await get_rewards(
        document=test_doc,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        responses=responses,
        client=AsyncOpenAI(),
        num_embeddings=150,
        reward_options=RewardOptions(),
        verbose=True,
    )

    logger.info(f"Rewards: {rewards}")

    assert rewards[0] != rewards[1]

    time_penalties = [extra_info["time_penalty"] for extra_info in extra_infos]

    logger.info(f"Time penalties: {time_penalties}")

    assert time_penalties[0] is None
    assert time_penalties[1] is not None

    logger.info("Identical responses with same process time should have same final reward")

    for response in responses:
        response.dendrite.process_time = 10

    rewards, extra_infos = await get_rewards(
        document=test_doc,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        responses=responses,
        client=AsyncOpenAI(),
        num_embeddings=150,
        reward_options=RewardOptions(),
        verbose=True,
    )

    logger.info(f"Rewards: {rewards}")

    assert rewards[0] == rewards[1]

    time_penalties = [extra_info["time_penalty"] for extra_info in extra_infos]

    logger.info(f"Time penalties: {time_penalties}")

    for time_penalty in time_penalties:
        assert time_penalty is None



def test_time_penalty():
    asyncio.run(main())
