from enum import Enum
from typing import List, Optional

from fastapi import Body
from pydantic import BaseModel, Field


class ChunkRequestType(Enum):
    normal = "normal"
    benchmark = "benchmark"


class RewardOptions(BaseModel):
    with_checks: bool = Body(
        default=True,
        description="Whether to check that chunks appropriately match the document",
    )
    with_penalties: bool = Body(
        default=True,
        description="Whether to apply penalties when grading chunks (e.g. size, quantity, time)",
    )


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
    request_type: ChunkRequestType = Body(
        default=ChunkRequestType.normal,
        description="The type of chunking task to run",
    )
    reward_options: RewardOptions = Body(
        default=RewardOptions(),
        description="Options for rewarding chunks",
    )
    benchmark_id: Optional[str] = Body(
        default=None,
        description="The associated benchmark id (if applicable)",
    )
    doc_name: Optional[str] = Body(
        default=None,
        description="The name of the document",
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
