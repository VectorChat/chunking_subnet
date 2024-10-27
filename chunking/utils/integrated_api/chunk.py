import traceback
from typing import Optional, List
from fastapi import Body
from pydantic import BaseModel, Field
from chunking.utils.chunks import calculate_chunk_qty
from chunking.utils.integrated_api.log import api_log
from chunking.utils.relay.relay import make_relay_payload
from chunking.validator.task_api import Task
from chunking.validator.tournament import run_tournament_round
from chunking.protocol import chunkSynapse
import bittensor as bt


class ChunkRequest(BaseModel):
    document: str = Body(..., description="The document to chunk")
    chunk_size: int = Body(
        ..., description="The maximum size of each chunk in characters", gt=0
    )
    chunk_qty: Optional[int] = Body(
        None,
        description="Max number of chunks to create, defaults to `ceil(ceil(len(document) / chunk_size) * 1.5)`",
        gt=0,
    )
    timeout: Optional[float] = Body(
        default=60, description="Hard timeout for the chunking task", gt=0
    )
    time_soft_max_multiplier: Optional[float] = Body(
        default=0.75,
        description="Soft max multiplier for the chunking task, defaults to 0.75 times timeout. Time after timeout * time_soft_max_multiplier is considered as a time penalty for the miner.",
    )
    custom_miner_uids: Optional[List[int]] = Body(
        default=None, description="Specific miner UIDs to query", min_length=1
    )
    miner_group_index: Optional[int] = Body(
        default=None, description="Specific miner group index to query", ge=0
    )
    do_scoring: bool = Body(
        default=False,
        description="Whether chunks should count towards scores in the tournament",
    )
    do_wandb_log: bool = Body(
        default=False,
        description="Whether to log the chunking task to wandb",
    )


class ChunkResult(BaseModel):
    chunks: List[str] = Field(
        ..., description="List of chunks resulting from the chunking task"
    )
    miner_signature: str = Field(
        ..., description="The signature of the miner that generated the chunks"
    )
    uid: int = Field(..., description="The UID of the miner that generated the chunks")
    miner_group_index: Optional[int] = Field(
        default=None,
        description="The index of the miner group that generated the chunks",
    )
    process_time: float = Field(
        ...,
        description="The time it took to process the chunking task (including network i/o)",
    )


class ChunkResponse(BaseModel):
    results: List[ChunkResult] = Field(
        ..., description="List of chunk results from the chunking task"
    )


async def chunk_handler(self, request: ChunkRequest) -> ChunkResponse:

    print(
        f"handling chunk request with document length: {len(request.document)}, miner uids: {request.custom_miner_uids}, miner group index: {request.miner_group_index}"
    )

    chunk_qty = request.chunk_qty or calculate_chunk_qty(
        request.document, request.chunk_size
    )

    input_synapse = chunkSynapse(
        document=request.document,
        chunk_size=request.chunk_size,
        chunk_qty=chunk_qty,
        timeout=request.timeout,
        time_soft_max=request.timeout * request.time_soft_max_multiplier,
    )

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

    results = await run_tournament_round(
        self,
        task=task,
        choose_miner_group_index=request.miner_group_index,
        custom_miner_uids=request.custom_miner_uids,
        do_wandb_log=request.do_wandb_log,
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
