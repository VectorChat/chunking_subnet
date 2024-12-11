import asyncio
import json
import logging
import random
import time
import httpx

from chunking.utils.integrated_api.chunk.types import ChunkResponse
from chunking.utils.synthetic.synthetic import get_wiki_content_for_page
from tests.utils.articles import get_articles


logger = logging.getLogger(__name__)
BASE_URL = "http://localhost:8080"


async def query_integrated_api(
    miner_group_index: int,
    document: str,
    do_wandb_log: bool,
    do_scoring: bool,
    chunk_size: int,
    custom_miner_uids: list[int] | None,
):
    async with httpx.AsyncClient() as client:
        req_body = {
            "miner_group_index": miner_group_index,
            "chunk_size": chunk_size,
            "document": document,
            "do_wandb_log": do_wandb_log,
            "do_scoring": do_scoring,
        }

        if custom_miner_uids is not None:
            req_body["custom_miner_uids"] = custom_miner_uids

        response = await client.post(
            f"{BASE_URL}/chunk",
            json=req_body,
            timeout=None,
        )

        return response


async def get_doc_and_query(
    pageid: int,
    miner_group_index: int,
    chunk_size: int = 4096,
    do_wandb_log: bool = True,
    do_scoring: bool = True,
    custom_miner_uids: list[int] | None = None,
):
    logger.info(f"Getting document for pageid {pageid}")
    content, title = await get_wiki_content_for_page(pageid)

    logger.info(f"Got document {title} of len {len(content)}")

    start_time = time.time()

    res = await query_integrated_api(
        miner_group_index=miner_group_index,
        document=content,
        do_wandb_log=do_wandb_log,
        do_scoring=do_scoring,
        chunk_size=chunk_size,
        custom_miner_uids=custom_miner_uids,
    )

    end_time = time.time()

    res_time = end_time - start_time

    logger.info(
        f"Got response for group {miner_group_index}, pageid {pageid} of doc of len {len(content)} in {res_time} seconds"
    )

    return res


async def main(num_articles: int, batch_size: int):

    test_pageids = get_articles()

    max_miner_group_index = 3

    batch_times = []

    for i in range(0, num_articles, batch_size):
        batch_pageids = test_pageids[i : i + batch_size]

        rand_miner_group_index = random.randint(0, max_miner_group_index)

        start_time = time.time()

        coros = [
            get_doc_and_query(pageid, rand_miner_group_index)
            for pageid in batch_pageids
        ]

        logger.info(f"Made {len(coros)} coroutines")

        responses = await asyncio.gather(*coros)

        end_time = time.time()

        batch_times.append(end_time - start_time)

        print(f"Responses received for batch {i} in {batch_times[-1]} seconds")

        logger.info(f"Got {len(responses)} responses")

        for response in responses:
            assert response is not None
            res_json = response.json()
            assert ChunkResponse.model_validate(res_json)

    print(f"Batch times: {json.dumps(batch_times, indent=2)}")


def test_integrated_api(num_articles, batch_size):
    asyncio.run(main(num_articles, batch_size))
