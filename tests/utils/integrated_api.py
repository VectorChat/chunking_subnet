import logging
import time
import httpx
from chunking.utils.integrated_api.chunk.types import ChunkRequestType
from chunking.utils.synthetic.synthetic import get_wiki_content_for_page


BASE_URL = "http://localhost:8080"
logger = logging.getLogger(__name__)


async def query_integrated_api(
    miner_group_index: int | None,
    document: str,
    do_wandb_log: bool,
    do_scoring: bool,
    chunk_size: int,
    custom_miner_uids: list[int] | None,
    type: ChunkRequestType,
    benchmark_id: str | None = None,
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
        }

        if miner_group_index is not None:
            req_body["miner_group_index"] = miner_group_index

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
    miner_group_index: int | None = None,
    chunk_size: int = 4096,
    do_wandb_log: bool = True,
    do_scoring: bool = True,
    custom_miner_uids: list[int] | None = None,
    type: ChunkRequestType = ChunkRequestType.normal,
    benchmark_id: str | None = None,
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
        type=type,
        benchmark_id=benchmark_id,
    )

    end_time = time.time()

    res_time = end_time - start_time

    logger.info(
        f"Got response for group {miner_group_index}, pageid {pageid} of doc of len {len(content)} in {res_time} seconds"
    )

    return res
