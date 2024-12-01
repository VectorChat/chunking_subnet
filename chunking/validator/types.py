from typing import Literal, Optional
import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel
from chunking.protocol import chunkSynapse


TaskType = Literal["organic", "synthetic"]


class EndTournamentRoundInfo(BaseModel):
    ranked_responses_global: NDArray[Shape["*"], float] 
    miner_group_uids: NDArray[Shape["*"], int]
    miner_group_index: int
    alpha: float
    responses: list[chunkSynapse]
    rewards: NDArray[Shape["*"], float]
    wandb_data: Optional[dict] = None
    do_wandb_log: bool = False
    task_type: TaskType
