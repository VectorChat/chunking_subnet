import argparse
from ast import arg
import asyncio
import json
import os
from openai import AsyncOpenAI
from chunking.utils.ipfs import add_to_ipfs_cluster
from chunking.utils.relay.types import RelayMessage, RelayPayload
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
        raise ValueError(
            "Either self or openai_client and embedding_model must be provided"
        )

    openai_client = self.aclient if self else openai_client
    embedding_model = self.embedding_model if self else embedding_model

    tokens = get_tokens_from_string(document, embedding_model)

    token_limit = target_token_amt

    embed_chunks = []
    for i in range(0, len(tokens), token_limit):
        chunk_tokens = tokens[i : i + token_limit]
        chunk = get_string_from_tokens(chunk_tokens, embedding_model)
        _verbose(f"Chunk: {chunk[:100]}...")
        embed_chunks.append(chunk)

    _verbose(
        f"Embed chunk sizes: {[num_tokens_from_string(chunk, embedding_model) for chunk in embed_chunks]}"
    )

    coros = []
    for chunk in embed_chunks:
        coros.append(
            openai_client.embeddings.create(model=embedding_model, input=chunk)
        )

    _verbose(f"Waiting for {len(coros)} coroutines to complete")

    results = await asyncio.gather(*coros)

    embeddings = [result.data[0].embedding for result in results]

    for i, embedding in enumerate(embeddings):
        _verbose(f"Embedding {i} size: {len(embedding)}")
        # if any NaN in embedding, print the embedding
        if np.any(np.isnan(embedding)):
            _verbose(f"Embedding {i} has NaN values: {embedding}\n\nCorresponding chunk: {embed_chunks[i][:100]}...")

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
        self, document, openai_client, embedding_model, verbose
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
