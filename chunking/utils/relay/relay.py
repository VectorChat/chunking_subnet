import argparse
from ast import arg
import asyncio
from datetime import timedelta
import json
import os
from typing import List
from openai import AsyncOpenAI
from chunking.utils.ipfs.ipfs import add_to_ipfs_cluster, get_from_ipfs, get_pinned_cids
from chunking.utils.relay.types import IPFSRelayPin, RelayMessage, RelayPayload
from chunking.utils.tokens import (
    get_string_from_tokens,
    get_tokens_from_string,
    num_tokens_from_string,
)
from neurons.validator import Validator
import hashlib
import bittensor as bt
import numpy as np


def sha256_hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


async def make_embeddings(
    self: Validator | None,
    document: str,
    openai_client: AsyncOpenAI | None = None,
    embedding_model: str | None = None,
    target_token_amt: int = 5000,
    verbose=False,
) -> str:
    def _verbose(message: str):
        if verbose:
            bt.logging.debug(message)

    _verbose(f"Making embeddings for document of length {len(document)} chars")

    if not self and (not openai_client and not embedding_model):
        _verbose(
            "No validator or OpenAI client or embedding model provided, using default"
        )
        openai_client = AsyncOpenAI()
        embedding_model = "text-embedding-ada-002"
        _verbose(
            f"Using OpenAI client: {openai_client} and embedding model: {embedding_model}"
        )
    else:
        openai_client = self.aclient if self else openai_client
        embedding_model = self.embedding_model if self else embedding_model
        _verbose(
            f"Using provided OpenAI client: {openai_client} and embedding model: {embedding_model}"
        )

    tokens = get_tokens_from_string(document, embedding_model)
    _verbose(f"Got {len(tokens)} tokens")

    token_limit = target_token_amt

    _verbose(f"Token limit: {token_limit}")

    embed_chunks = []
    for i in range(0, len(tokens), token_limit):
        chunk_tokens = tokens[i : i + token_limit]
        chunk = get_string_from_tokens(chunk_tokens, embedding_model)
        _verbose(f"Chunk: {chunk[:100]}...")
        embed_chunks.append(chunk)

    _verbose(
        f"Embed chunk sizes: {[num_tokens_from_string(chunk, embedding_model) for chunk in embed_chunks]}"
    )

    async def get_embedding(chunk: str, i: int) -> list[float]:
        try:
            _verbose(f"Getting embedding for chunk {i}")
            result = await openai_client.embeddings.create(
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
    self: Validator | None,
    document: str,
    openai_client: AsyncOpenAI | None = None,
    embedding_model: str | None = None,
    wallet: bt.wallet | None = None,
    verbose=False,
) -> str:

    def _verbose(message: str):
        if verbose:
            bt.logging.debug(message)

    _verbose(f"Making relay payload for document of length {len(document)} chars")

    doc_hash = sha256_hash(document)
    _verbose(f"Document hash: {doc_hash}")

    embeddings = await make_embeddings(
        self, document, openai_client, embedding_model, verbose=verbose
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

    wallet = self.wallet if self else wallet
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

    tmp_file = "tmp_relay_payload.json"

    with open(tmp_file, "w") as f:
        json.dump(payload_dict, f)
        _verbose(f"Wrote IPFS payload to {tmp_file}")

    cid = add_to_ipfs_cluster(tmp_file)

    _verbose(f"Added IPFS payload to cluster with CID: {cid}")

    os.remove(tmp_file)

    _verbose(f"Removed temporary file {tmp_file}")

    return cid


async def get_relay_payload(cid: str, verbose=False) -> RelayPayload:
    raw_content = await get_from_ipfs(cid, verbose=verbose)
    obj = json.loads(raw_content)
    payload = RelayPayload(**obj)
    return payload


async def get_recent_relay_pins(
    delta=timedelta(minutes=20), verbose=False
) -> List[IPFSRelayPin]:
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

    client = AsyncOpenAI()
    cid = asyncio.run(
        make_relay_payload(
            None, document, client, args.embedding_model, wallet, verbose=True
        )
    )
    print(f"CID: {cid}")