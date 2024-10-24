import argparse
import asyncio
import random
from re import A

from openai import AsyncOpenAI, OpenAI
from chunking.protocol import chunkSynapse
from chunking.utils.chunks import calculate_chunk_qty
from chunking.utils.relay.relay import make_relay_payload
from chunking.validator.task_api import (
    Task,
    generate_doc_normal,
    get_wiki_content_for_page,
)
import bittensor as bt
from tests.utils.articles import get_articles


async def runner(args: argparse.Namespace):
    chain_endpoint = args.chain_endpoint
    axon_wallet = args.axon_wallet
    axon_hotkey = args.axon_hotkey
    axon_ip = args.axon_ip
    axon_port = args.axon_port

    articles = get_articles()

    bt.logging.set_debug()

    bt.logging.debug(f"Got {len(articles)} articles")

    COLDKEY = "owner-localnet"
    HOTKEY = "validator1"

    vali_wallet = bt.wallet(name=COLDKEY, hotkey=HOTKEY)

    vali_dendrite = bt.dendrite(wallet=vali_wallet)

    client = OpenAI()
    aclient = AsyncOpenAI()

    # doc = generate_doc_with_llm(None, pageids=pageids, timeout=20, client=client)

    random_article_index = random.randint(0, len(articles) - 1)
    bt.logging.debug(f"Random article index: {random_article_index}")

    random_article = articles[random_article_index]

    bt.logging.debug(f"Random article: {random_article}")

    doc, title = get_wiki_content_for_page(random_article)

    # with open("test_doc.txt") as f:
    #     doc = f.read()

    bt.logging.debug(f"Generated doc, {title}, {len(doc)} chars")

    if args.no_cid:
        cid = None
    elif args.fake_cid:
        cid = "FAKECID"
    else:
        bt.logging.debug("Making relay payload")
        cid = await make_relay_payload(
            doc, aclient, vali_wallet, "text-embedding-ada-002", True
        )

    bt.logging.debug(f"Using CID: {cid}")

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

    print(f"Made synapse, keys: {synapse.__dict__.keys()}")

    if args.use_local_axon:
        axon_wallet = bt.wallet(name=axon_wallet, hotkey=axon_hotkey)

        axon = bt.axon(
            wallet=axon_wallet,
            ip=axon_ip,
            port=axon_port,
        )
    else:
        metagraph = bt.metagraph(netuid=1, network=chain_endpoint)
        uid = args.uid
        axon = metagraph.axons[uid]

    axons = [axon]

    bt.logging.debug(f"Querying axons: {axons}")

    responses: list[chunkSynapse] = vali_dendrite.query(
        axons=axons,
        timeout=synapse.timeout,
        synapse=synapse,
        deserialize=False,
    )

    chunks = responses[0].chunks
    bt.logging.debug(
        f"Received {len(chunks)} chunks" if chunks else "No chunks received"
    )


def test_relay_round(args: argparse.Namespace):
    asyncio.run(runner(args))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--chain_endpoint", type=str, default="ws://127.0.0.1:9946")
    argparser.add_argument("--axon_wallet", type=str, default="owner-localnet")
    argparser.add_argument("--axon_hotkey", type=str, default="miner1")
    argparser.add_argument("--axon_ip", type=str, default="127.0.0.1")
    argparser.add_argument("--axon_port", type=int, default=8092)
    argparser.add_argument("--use_local_axon", action="store_true")
    argparser.add_argument("--no_cid", action="store_true")
    argparser.add_argument("--fake_cid", action="store_true")
    argparser.add_argument("--uid", type=int, default=16)
    args = argparser.parse_args()
    test_relay_round(args)
