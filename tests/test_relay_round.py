import argparse
import asyncio
from random import random
from re import A

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


async def runner(chain_endpoint: str):
    articles = get_articles()

    bt.logging.set_debug()

    bt.logging.debug(f"Got {len(articles)} articles")

    pageids = articles[:3]

    COLDKEY = "owner-localnet"
    HOTKEY = "validator1"

    vali_wallet = bt.wallet(name=COLDKEY, hotkey=HOTKEY)

    vali_dendrite = bt.dendrite(wallet=vali_wallet)

    client = OpenAI()
    aclient = AsyncOpenAI()

    # doc = generate_doc_with_llm(None, pageids=pageids, timeout=20, client=client)

    with open("test_doc.txt") as f:
        doc = f.read()

    bt.logging.debug(f"Generated doc, {len(doc)} chars")

    cid = await make_relay_payload(
        None, doc, aclient, "text-embedding-ada-002", vali_wallet
    )

    bt.logging.debug(f"Made relay payload: {cid}")

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

    metagraph = bt.metagraph(netuid=1, network=chain_endpoint)

    test_uid = 16

    axons: list[bt.axon] = [metagraph.axons[test_uid]]

    bt.logging.debug(f"Querying axons: {axons}")

    responses: list[chunkSynapse] = vali_dendrite.query(
        axons=axons,
        timeout=synapse.timeout,
        synapse=synapse,
        deserialize=False,
    )

    bt.logging.debug(f"Received responses: {responses}")

    chunks = responses[0].chunks
    bt.logging.debug(f"Received {len(chunks)} chunks" if chunks else "No chunks received")


def test_relay_round(chain_endpoint: str):
    asyncio.run(runner(chain_endpoint))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--chain_endpoint", type=str, default="ws://127.0.0.1:9946")
    args = argparser.parse_args()
    test_relay_round(args.chain_endpoint)
