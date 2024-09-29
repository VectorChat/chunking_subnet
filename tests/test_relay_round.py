import asyncio
from random import random

from openai import AsyncOpenAI, OpenAI
from chunking.protocol import chunkSynapse
from chunking.validator.relay import make_relay_payload
from chunking.validator.task_api import (
    Task,
    calculate_chunk_qty,
    generate_doc_with_llm,
    generate_synthetic_synapse,
)
import bittensor as bt

from tests.utils.articles import get_articles


async def runner():
    articles = get_articles()

    bt.logging.set_debug()

    print(f"Got {len(articles)} articles")

    pageids = articles[:3]

    COLDKEY = "owner-localnet"
    HOTKEY = "validator2"

    vali_wallet = bt.wallet(name=COLDKEY, hotkey=HOTKEY)

    vali_dendrite = bt.dendrite(wallet=vali_wallet)

    client = OpenAI()
    aclient = AsyncOpenAI()

    doc = generate_doc_with_llm(None, pageids=pageids, timeout=20, client=client)

    print(f"Generated doc: {doc}")

    cid = await make_relay_payload(None, doc, client, "text-embedding-ada-002", vali_wallet)

    print(f"Made relay payload: {cid}")

    chunk_size = 4096
    timeout = 20

    synapse = chunkSynapse(
        document=doc,
        chunk_size=chunk_size,
        chunk_qty=calculate_chunk_qty(doc, chunk_size),
        timeout=timeout,
        time_soft_max=timeout * 0.75,
        CID=cid,
    )

    metagraph = bt.metagraph(netuid=1, network="ws://127.0.0.1:9946")

    test_uid = 16

    axons: list[bt.axon] = [metagraph.axons[test_uid]]

    responses: list[chunkSynapse] = vali_dendrite.query(
        axons=axons,
        timeout=synapse.timeout,
        synapse=synapse,
        deserialize=False,
    )

    print(f"Received responses: {responses}")

    assert len(responses) == 1
    assert responses[0].chunks is not None


def test_relay_round():
    asyncio.run(runner())


if __name__ == "__main__":
    test_relay_round()
