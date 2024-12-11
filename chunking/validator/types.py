from typing import Literal, Optional
import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel
from chunking.protocol import chunkSynapse


TaskType = Literal["organic", "synthetic"]


class EndTournamentRoundInfo(BaseModel):
    """
    Information about the end of a tournament round, used to update scores
    """

    # rank values, "effective" rank that should be used when updating scores
    rank_values: NDArray[Shape["*"], float]
    # UIDs of miners in the group
    miner_group_uids: NDArray[Shape["*"], int]
    # index of the miner group
    miner_group_index: int
    # alpha value for the group
    alpha: float
    # best possible rank value for a miner group, not necessarily the best rank value that a miner actually achieved in the group for the round
    group_best_possible_rank_value: float
    # responses to the task
    responses: list[chunkSynapse]
    # rewards for the responses
    rewards: NDArray[Shape["*"], float]
    # for logging to wandb
    wandb_data: Optional[dict] = None
    # whether to log to wandb
    do_wandb_log: bool = False
    # type of task
    task_type: TaskType
