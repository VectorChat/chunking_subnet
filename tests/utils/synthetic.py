import asyncio
import os
import random
from typing import List, Tuple

from openai import AsyncOpenAI

from chunking.utils.synthetic import generate_doc_with_llm


async def generate_and_save_synthetic_document(
    all_pageids: List[int], k: int, loop_range: range, aclient: AsyncOpenAI
):
    pageids = random.sample(all_pageids, k=k)

    synthetic_document, article_names = await generate_doc_with_llm(
        validator=None,
        pageids=pageids,
        k=k,
        loop_range=loop_range,
        override_client=aclient,
    )

    file_name = "_".join(article_names).replace(" ", "_")

    file_name = f"{file_name}_k={k}_r={loop_range.start}-{loop_range.stop}.txt"

    save_path = f"tests/assets/synthetic_documents/{file_name}.txt"

    with open(save_path, "w") as f:
        f.write(synthetic_document)
        print(f"Saved synthetic document to {save_path}")

    return synthetic_document

async def generate_and_save_synthetic_data(
    n: int,
    all_pageids: List[int],
    k: int,
    loop_range: range,
    aclient: AsyncOpenAI,
    synth_gen_batch_size: int,
):

    all_synthetic_documents = []

    for i in range(0, n, synth_gen_batch_size):
        coros = []

        for _ in range(synth_gen_batch_size):
            coros.append(
                generate_and_save_synthetic_document(
                    all_pageids, k, loop_range, aclient
                )
            )

        all_synthetic_documents.extend(await asyncio.gather(*coros))

    return all_synthetic_documents


def get_save_dir():
    return "tests/assets/synthetic_documents"


def load_synthetic_data(n: int):
    save_dir = get_save_dir()

    files = os.listdir(save_dir)

    for file in files[:n]:
        save_path = os.path.join(save_dir, file)
        with open(save_path, "r") as f:
            yield f.read(), save_path


def get_num_saved_synthetic_documents():
    save_dir = get_save_dir()

    return len(os.listdir(save_dir))


async def get_or_load_synthetic_data(
    n: int,
    all_pageids: List[int],
    k: int,
    loop_range: range,
    aclient: AsyncOpenAI,
    synth_gen_batch_size: int,
) -> List[Tuple[str, str]]:
    num_cached_documents = get_num_saved_synthetic_documents()

    if num_cached_documents < n:
        await generate_and_save_synthetic_data(
            n - num_cached_documents,
            all_pageids,
            k,
            loop_range,
            aclient,
            synth_gen_batch_size,
        )

    return list(load_synthetic_data(n))
