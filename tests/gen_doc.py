import argparse
import asyncio
import os
import random
import re
from typing import Literal
import bittensor as bt
from openai import AsyncOpenAI
from chunking.validator.task_api import generate_doc_with_llm
from tests.utils.articles import get_articles

argparser = argparse.ArgumentParser()

argparser.add_argument("--temperature", "-t", type=float, default=0.7)
argparser.add_argument("--k", "-k", type=int, default=3)
argparser.add_argument("--num_iters", "-n", type=int, default=10)
argparser.add_argument("--loop_range", "-l", nargs="+", type=int)
argparser.add_argument("--concurrent_n", "-cn", type=int, default=3)
argparser.add_argument("--gen_types", "-gt", nargs="+", type=str)


async def gen_and_save_doc(args, articles_sample, gen_type: Literal["old", "new"], loop_range):

    document, article_names = await generate_doc_with_llm(
        validator=None,
        pageids=articles_sample,
        temperature=args.temperature,
        k=args.k,
        loop_range=loop_range,
        override_client=AsyncOpenAI(),
        gen_type=gen_type,
    )

    file_path = f"tests/assets/documents/{gen_type.upper()}_"
    clean_regex = r"[\\/:*?\"<>|]"
    for article_name in article_names:
        cleaned_name = re.sub(clean_regex, "", article_name)
        space_to_hyphen = re.sub(r" ", "-", cleaned_name)
        file_path += f"{space_to_hyphen}_"
    file_path += f"t-{args.temperature}_k-{args.k}.txt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    print(f"Saving document to {file_path}")
    with open(file_path, "w") as f:
        f.write(document)
    print(f"Saved document to {file_path}")


async def main():
    args = argparser.parse_args()
    print(args)

    bt.debug()

    articles = get_articles()
    print(f"Found {len(articles)} articles")

    loop_range = args.loop_range
    if loop_range is None or len(loop_range) == 0:
        loop_range = range(3, 7)
    elif len(loop_range) != 2:
        raise ValueError("Loop range must be a pair of integers")

    if loop_range[0] >= loop_range[1]:
        raise ValueError("Loop range start must be less than end")

    loop_range = range(loop_range[0], loop_range[1])

    print(f"loop_range: {loop_range}")

    gen_types = args.gen_types
    if gen_types is None or len(gen_types) == 0:
        gen_types = ["old", "new"]

    for i in range(args.num_iters):
        print(f"outer iter {i}")

        coros = []

        for _ in range(args.concurrent_n):
            articles_sample = random.sample(articles, args.k)
            print(f"articles_sample: {articles_sample}")
            for gen_type in gen_types:
                print(f"gen_type: {gen_type}")
                coros.append(gen_and_save_doc(args, articles_sample, gen_type, loop_range))

        await asyncio.gather(*coros)


if __name__ == "__main__":
    asyncio.run(main())
