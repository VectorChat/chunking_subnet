import argparse
import asyncio
import os
from typing import List
from openai import AsyncOpenAI
import pandas as pd
import regex as re
import bittensor as bt
from substrateinterface import Keypair

from chunking.protocol import chunkSynapse
from chunking.validator.reward import get_rewards

argparser = argparse.ArgumentParser()
argparser.add_argument("--round_dir", "-r", type=str, required=True)


def read_chunks_from_csv(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    return df["Chunk Content"].tolist()


def get_uid_from_csv_path(csv_path: str) -> str:
    pattern = r"uid_(\d+).csv"
    match = re.search(pattern, csv_path)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract uid from {csv_path}")

def random_ss58_address() -> str:
    mnemonic = Keypair.generate_mnemonic()

    keypair = Keypair.create_from_mnemonic(mnemonic)
    address = keypair.ss58_address
    return address

def make_response(
    source_doc: str, chunks: List[str], chunk_size: int, chunk_qty: int
) -> chunkSynapse:
    synapse = chunkSynapse(
        document=source_doc,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        chunks=chunks,
        time_soft_max=15.0,
        timeout=20.0,
    )

    synapse.axon = bt.TerminalInfo(
        ip="127.0.0.1",
        port=8090,
        hotkey=random_ss58_address(),
    )

    return synapse


async def main():
    args = argparser.parse_args()
    round_dir = args.round_dir

    source_doc_path = os.path.join(round_dir, "doc.txt")
    with open(source_doc_path, "r") as f:
        source_doc = f.read()

    csvs_dir = os.path.join(round_dir, "csvs")
    csv_paths = [os.path.join(csvs_dir, f) for f in os.listdir(csvs_dir)]
    uids = [get_uid_from_csv_path(csv_path) for csv_path in csv_paths]
    chunks = [read_chunks_from_csv(csv_path) for csv_path in csv_paths]

    for uid, chunk in zip(uids, chunks):
        print(f"UID: {uid} made {len(chunk)} chunks")

    responses = [make_response(source_doc, chunk, 3000, 26) for chunk in chunks]
    rewards, extra_infos = await get_rewards(
        source_doc,
        3000,
        26,
        responses,
        client=AsyncOpenAI(),
        num_embeddings=150,
    )

    for uid, reward, extra_info in zip(uids, rewards, extra_infos):
        print(f"UID: {uid} reward: {reward}")


if __name__ == "__main__":
    asyncio.run(main())
