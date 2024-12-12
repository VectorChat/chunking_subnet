from enum import Enum
import traceback
from typing import Optional, List
from fastapi import Body
from pydantic import BaseModel, Field
from chunking.utils.chunks import calculate_chunk_qty
from chunking.utils.integrated_api.chunk.types import (
    ChunkRequest,
    ChunkRequestType,
    ChunkResponse,
    ChunkResult,
)
from chunking.utils.integrated_api.log import api_log
from chunking.utils.relay.relay import make_relay_payload
from chunking.validator.task_api import Task
from chunking.validator.tournament import run_tournament_round
from chunking.protocol import chunkSynapse
import bittensor as bt


async def chunk_handler(self, request: ChunkRequest) -> ChunkResponse:

    if request.request_type == ChunkRequestType.benchmark:
        if not request.benchmark_id:
            raise ValueError("Benchmark id is required for benchmark requests")

    print(
        f"handling chunk request with document length: {len(request.document)}, miner uids: {request.custom_miner_uids}, miner group index: {request.miner_group_index}"
    )

    chunk_qty = request.chunk_qty or calculate_chunk_qty(
        request.document, request.chunk_size
    )

    bt.logging.info(f"chunk qty: {chunk_qty}")

    input_synapse = chunkSynapse(
        document=request.document,
        chunk_size=request.chunk_size,
        chunk_qty=chunk_qty,
        timeout=request.timeout,
        time_soft_max=request.timeout * request.time_soft_max_multiplier,
    )

    if request.do_scoring:
        try:
            CID = await make_relay_payload(
                input_synapse.document, self.aclient, self.wallet
            )
        except Exception as e:
            api_log(f"Error making relay payload: {e}")
            traceback.print_exc()
            CID = None

        input_synapse.CID = CID

    task = Task(synapse=input_synapse, task_type="organic", task_id=-1, page_id=-1)

    bt.logging.info("created task")

    results = await run_tournament_round(
        self,
        task=task,
        choose_miner_group_index=request.miner_group_index,
        custom_miner_uids=request.custom_miner_uids,
        do_wandb_log=request.do_wandb_log,
        request_type=request.request_type,
        reward_options=request.reward_options,
        benchmark_id=request.benchmark_id,
        doc_name=request.doc_name,
    )

    usable_results = [result for result in results if result is not None]

    api_log(f"Got {len(usable_results)} usable results out of {len(results)}")

    # TODO: make background task if possible
    #
    chunk_results: List[ChunkResult] = []
    for result in usable_results:
        if request.custom_miner_uids is not None and request.do_wandb_log:
            # manually log because it will not be logged during score update
            bt.logging.debug(
                f"Logging wandb data for custom miner uids: {request.custom_miner_uids}"
            )
            self.wandb_log(result.wandb_data)
        elif request.do_scoring:
            await self.queue_score_update(result)

        miner_group_uids = result.miner_group_uids

        for response, miner_group_uid in zip(result.responses, miner_group_uids):
            if response.chunks and response.miner_signature:
                api_log(
                    f"Got {len(response.chunks)} chunks from miner {miner_group_uid} in group {result.miner_group_index}"
                )
                chunk_results.append(
                    ChunkResult(
                        chunks=response.chunks,
                        miner_signature=response.miner_signature,
                        uid=miner_group_uid,
                        miner_group_index=result.miner_group_index,
                        process_time=response.dendrite.process_time,
                    )
                )

    return ChunkResponse(results=chunk_results)
