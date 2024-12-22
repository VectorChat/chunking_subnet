import asyncio
import json
import logging
import random
import time
import httpx

from chunking.utils.integrated_api.chunk.types import ChunkRequestType, ChunkResponse
from chunking.utils.synthetic.synthetic import get_wiki_content_for_page
from tests.utils.articles import get_articles
import bittensor as bt

from tests.utils.misc import get_abbreviated_dict_string


logger = logging.getLogger(__name__)
BASE_URL = "http://localhost:8080"


async def query_integrated_api(
    miner_group_index: int | None,
    document: str,
    do_wandb_log: bool,
    do_scoring: bool,
    chunk_size: int,
    custom_miner_uids: list[int] | None,
    type: ChunkRequestType,
    benchmark_id: str | None = None,
    timeout: float = 20,
    time_soft_max_multiplier: float = 0.75,
):
    if not miner_group_index and not custom_miner_uids:
        raise ValueError(
            "Either miner_group_index or custom_miner_uids must be provided"
        )

    async with httpx.AsyncClient() as client:
        req_body = {
            "chunk_size": chunk_size,
            "document": document,
            "do_wandb_log": do_wandb_log,
            "do_scoring": do_scoring,
            "request_type": type.value,
            "benchmark_id": benchmark_id,
            "timeout": timeout,
            "time_soft_max_multiplier": time_soft_max_multiplier,
        }

        logger.info(f"Req body:\n{get_abbreviated_dict_string(req_body)}")

        if miner_group_index is not None:
            req_body["miner_group_index"] = miner_group_index

        if custom_miner_uids is not None:
            req_body["custom_miner_uids"] = custom_miner_uids

        req_url = f"{BASE_URL}/chunk"

        logger.info(f"Req url: {req_url}")

        response = await client.post(
            req_url,
            json=req_body,
            timeout=None,
        )

        return response


async def get_doc_and_query(
    pageid: int,
    miner_group_index: int | None = None,
    chunk_size: int = 4096,
    do_wandb_log: bool = True,
    do_scoring: bool = True,
    custom_miner_uids: list[int] | None = None,
    type: ChunkRequestType = ChunkRequestType.normal,
    benchmark_id: str | None = None,
    timeout: float = 20,
    time_soft_max_multiplier: float = 0.75,
):
    logger.info(
        f"Getting document for pageid {pageid}, group index: {miner_group_index}, custom uids: {custom_miner_uids}"
    )
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
        type=type,
        benchmark_id=benchmark_id,
        timeout=timeout,
        time_soft_max_multiplier=time_soft_max_multiplier,
    )

    end_time = time.time()

    res_time = end_time - start_time

    logger.info(
        f"Got response for group {miner_group_index}, pageid {pageid} of doc of len {len(content)} in {res_time} seconds"
    )

    return res


async def main(num_articles: int, batch_size: int):
    bt.debug()

    print("Getting test pageids")
    test_pageids = get_articles()

    print(f"Got {len(test_pageids)} test pageids")

    # test custom uids
    custom_uids = [16, 17, 18]

    rand_page_id = random.choice(test_pageids)

    res = await get_doc_and_query(
        pageid=rand_page_id,
        custom_miner_uids=custom_uids,
        type=ChunkRequestType.normal,
    )

    assert res is not None
    assert res.status_code == 200
    assert res.json() is not None
    res_json = res.json()
    res = ChunkResponse.model_validate(res_json)

    assert len(res.results) == len(custom_uids)

    for result in res.results:
        assert result.uid in custom_uids
        assert result.chunks is not None
        assert len(result.chunks) > 0

    logger.info("can query with custom uids")

    # test random miner group index

    max_miner_group_index = 2

    batch_times = []

    for i in range(0, num_articles, batch_size):
        batch_pageids = test_pageids[i : i + batch_size]

        rand_miner_group_index = random.randint(0, max_miner_group_index)

        start_time = time.time()

        coros = [
            get_doc_and_query(
                pageid=pageid,
                miner_group_index=rand_miner_group_index,
            )
            for pageid in batch_pageids
        ]

        logger.info(f"Made {len(coros)} coroutines")

        responses = await asyncio.gather(*coros)

        end_time = time.time()

        batch_times.append(end_time - start_time)

        logger.info(f"Responses received for batch {i} in {batch_times[-1]} seconds")

        logger.info(f"Got {len(responses)} responses")

        for i, response in enumerate(responses):
            assert response is not None
            res_json = response.json()
            logger.info(f"Response {i}: {res_json}")
            assert ChunkResponse.model_validate(res_json)

    logger.info(f"Batch times: {json.dumps(batch_times, indent=2)}")

    logger.info("can query with random miner group index")

    # test benchmark round

    rand_page_id = random.choice(test_pageids)

    miner_uids = [16, 17, 18]

    benchmark_id = "test_benchmark_id"

    res = await get_doc_and_query(
        pageid=rand_page_id,
        custom_miner_uids=miner_uids,
        type=ChunkRequestType.benchmark,
        benchmark_id=benchmark_id,
    )

    logger.info(f"Benchmark response: {res}")

    assert res is not None
    assert res.status_code == 200
    assert res.json() is not None
    res_json = res.json()
    assert ChunkResponse.model_validate(res_json)

    res = ChunkResponse.model_validate(res_json)

    assert len(res.results) == len(miner_uids)

    for result in res.results:
        assert result.uid in miner_uids
        assert result.chunks is not None
        assert len(result.chunks) > 0

    logger.info("can query with custom uids for benchmark round")

    # if no benchmark id is sent, should error

    res = await get_doc_and_query(
        pageid=rand_page_id,
        custom_miner_uids=miner_uids,
        type=ChunkRequestType.benchmark,
    )

    assert res is not None
    assert res.status_code == 500


def test_integrated_api(num_articles, batch_size):
    asyncio.run(main(num_articles, batch_size))


if __name__ == "__main__":
    print("Starting main")
    logger.setLevel(logging.DEBUG)
    asyncio.run(main(100, 1))
