import argparse
import asyncio
from datetime import datetime, timedelta
import json
import os
from tempfile import NamedTemporaryFile
import time
from typing import List
from openai import AsyncOpenAI
from chunking.utils.ipfs.ipfs import (
    add_to_ipfs_and_pin_to_cluster,
    get_from_ipfs,
    get_pinned_cids,
)
from chunking.utils.relay.types import IPFSRelayPin, RelayMessage, RelayPayload
from chunking.utils.tokens import (
    get_string_from_tokens,
    get_tokens_from_string,
    num_tokens_from_string,
)
import hashlib
import bittensor as bt
import numpy as np


def sha256_hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def get_embed_chunks(
    document: str, embedding_model: str, target_token_amt: int
) -> List[str]:
    """
    Splits the document into chunks that are then passed to the embedding model.

    Chunks are made deterministically by splitting the document into tokens, and then
    grouping the tokens into chunks of the target token amount. Split points are chosen
    at sentence boundaries.

    Args:
        document (str): The document to split into chunks.
        embedding_model (str): The embedding model to use.
        target_token_amt (int): The target token amount for each chunk.

    Returns:
        List[str]: The chunks of the document.
    """
    tokens = get_tokens_from_string(document, embedding_model)
    token_limit = target_token_amt
    embed_chunks = []
    for i in range(0, len(tokens), token_limit):
        chunk_tokens = tokens[i : i + token_limit]
        chunk = get_string_from_tokens(chunk_tokens, embedding_model)
        embed_chunks.append(chunk)
    return embed_chunks


async def make_embeddings(
    document: str,
    async_openai_client: AsyncOpenAI,
    embedding_model: str = "text-embedding-ada-002",
    target_token_amt: int = 5000,
    verbose=False,
) -> List[List[float]]:
    """
    Makes embeddings for the document for use by the miner to check for fuzzy duplicates.

    Args:
        document (str): The document to make embeddings for.
        async_openai_client (AsyncOpenAI): The OpenAI client to use.
        embedding_model (str): The embedding model to use.
        target_token_amt (int): The target token amount for each chunk.
        verbose (bool): Whether to print debug information.

    Returns:
        List[list[float]]: The embeddings for the document.
    """

    def _verbose(message: str):
        if verbose:
            bt.logging.debug(message)

    _verbose(f"Making embeddings for document of length {len(document)} chars")

    embed_chunks = get_embed_chunks(document, embedding_model, target_token_amt)

    _verbose(
        f"Embed chunk sizes: {[num_tokens_from_string(chunk, embedding_model) for chunk in embed_chunks]}"
    )

    async def get_embedding(chunk: str, i: int) -> list[float]:
        """
        Helper function to get the embedding for a chunk.

        Args:
            chunk (str): The chunk to make an embedding for.
            i (int): The index of the chunk.

        Returns:
            list[float]: The embedding for the chunk.
        """
        try:
            _verbose(f"Getting embedding for chunk {i}")
            result = await async_openai_client.embeddings.create(
                model=embedding_model, input=chunk
            )
            _verbose(f"Got embedding for chunk {i}")
            return result.data[0].embedding
        except Exception as e:
            _verbose(f"Error getting embedding for chunk {i}: {e}")
            return None

    coros = []
    for i, chunk in enumerate(embed_chunks):
        coros.append(get_embedding(chunk, i))

    _verbose(f"Waiting for {len(coros)} coroutines to complete")

    results = await asyncio.gather(*coros)

    embeddings = [result for result in results if result is not None]

    for i, embedding in enumerate(embeddings):
        _verbose(f"Embedding {i} size: {len(embedding)}")
        # if any NaN in embedding, print the embedding
        if np.any(np.isnan(embedding)):
            _verbose(
                f"Embedding {i} has NaN values: {embedding}\n\nCorresponding chunk: {embed_chunks[i][:100]}..."
            )

    return embeddings


async def make_relay_payload(
    document: str,
    openai_client: AsyncOpenAI,
    wallet: bt.wallet,
    embedding_model: str = "text-embedding-ada-002",
    verbose=False,
) -> str:
    """
    Makes a relay payload for the document for use by the miner to use to deter relay mining.
    The miner will use the document hash to check if the document has already been relayed.
    The miner will use the embeddings to check for fuzzy duplicates.

    Args:
        document (str): The document to make a relay payload for.
        openai_client (AsyncOpenAI): The OpenAI client to use.
        wallet (bt.wallet): The wallet to use.
        embedding_model (str): The embedding model to use.
        verbose (bool): Whether to print debug information.

    Returns:
        str: The CID of the relay payload.
    """

    def _verbose(message: str):
        if verbose:
            bt.logging.debug(message)

    _verbose(f"Making relay payload for document of length {len(document)} chars")

    doc_hash = sha256_hash(document)
    _verbose(f"Document hash: {doc_hash}")

    embeddings = await make_embeddings(
        document, openai_client, embedding_model, verbose=verbose
    )
    _verbose(f"Made {len(embeddings)} embeddings")

    message = RelayMessage(
        document_hash=doc_hash,
        embeddings=embeddings,
    )

    message_dict = message.model_dump()

    _verbose(f"Message: {json.dumps(message_dict)[:100]}...")

    message_hash = sha256_hash(json.dumps(message_dict))

    _verbose(f"Message hash: {message_hash}")

    message_sig = wallet.hotkey.sign(message_hash.encode()).hex()

    _verbose(f"Message signature: {message_sig}")

    payload = RelayPayload(
        message=message,
        signature=message_sig,
    )

    payload_dict = payload.model_dump()

    _verbose(
        f"IPFS payload:\n message: {json.dumps(payload_dict['message'])[:100]}...\n signature: {payload_dict['signature']}"
    )

    with NamedTemporaryFile(mode="w", suffix=".json", delete=True) as tmp_file:
        json.dump(payload_dict, tmp_file)
        tmp_file.flush()

        _verbose(f"Wrote IPFS payload to {tmp_file.name}")
        _verbose(f"Temp file size: {os.path.getsize(tmp_file.name)} bytes")

        start_time = time.time()
        cid = await add_to_ipfs_and_pin_to_cluster(
            tmp_file.name, expiry_delta=timedelta(minutes=3), verbose=verbose
        )

        bt.logging.info(
            f"Added IPFS payload to cluster with CID: {cid} in {time.time() - start_time} seconds"
        )

        if not cid:
            raise Exception("Failed to add IPFS payload to cluster")

    return cid


async def get_relay_payload(cid: str, verbose=False) -> RelayPayload:
    """
    Gets content from IPFS and parses it as a `RelayPayload`.

    Args:
        cid (str): The CID of the relay payload.
        verbose (bool): Whether to print debug information.

    Returns:
        RelayPayload: The relay payload.
    """
    raw_content = await get_from_ipfs(cid, verbose=verbose)
    obj = json.loads(raw_content)
    payload = RelayPayload(**obj)
    return payload


async def get_recent_relay_pins(
    delta=timedelta(minutes=20), verbose=False
) -> List[IPFSRelayPin]:
    """
    Gets recent relay pins from the IPFS cluster.

    It fetches all current pins and filters by the time delta.

    Args:
        delta (timedelta): The time delta to get pins for.
        verbose (bool): Whether to print debug information.

    Returns:
        List[IPFSRelayPin]: The recent relay pins.
    """

    def _verbose(message: str):
        if verbose:
            bt.logging.debug(message)

    _verbose(f"Getting recent relay pins with delta {delta}")

    pins = await get_pinned_cids(delta=delta)

    pins_with_payload = []

    for pin in pins:
        try:
            obj = json.loads(pin.raw_content)
            payload = RelayPayload(**obj)
            pin = IPFSRelayPin(
                cid=pin.cid,
                created_at=pin.created_at,
                payload=payload,
                raw_content=pin.raw_content,
            )
            pins_with_payload.append(pin)
        except Exception as e:
            _verbose(f"Error parsing payload for {pin.cid}: {e}")
            continue

    return pins_with_payload


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--in_file", type=str, required=True)
    argparser.add_argument(
        "--embedding_model", type=str, required=False, default="text-embedding-ada-002"
    )
    argparser.add_argument("--coldkey", type=str, default="owner-localnet")
    argparser.add_argument("--hotkey", type=str, default="validator1")

    args = argparser.parse_args()

    with open(args.in_file, "r") as f:
        document = f.read()

    wallet = bt.wallet(args.coldkey, args.hotkey)

    bt.debug()

    client = AsyncOpenAI()
    cid = asyncio.run(
        make_relay_payload(document, client, wallet, args.embedding_model, verbose=True)
    )
    print(f"CID: {cid}")
