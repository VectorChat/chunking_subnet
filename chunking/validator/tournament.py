import asyncio
import atexit
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging
import multiprocessing
import random
import sys
import threading
import traceback
import bittensor as bt
import numpy as np
from random import choice

from openai import AsyncOpenAI
from chunking.utils.integrated_api.chunk.types import ChunkRequestType, RewardOptions
from chunking.utils.log import PrefixStream
from chunking.utils.tournament import make_wandb_data, pretty_print_rewards
from chunking.validator.reward import get_rewards, rank_responses, rank_responses_global
from chunking.validator.task_api import Task
from chunking.validator.types import EndTournamentRoundInfo
from chunking.protocol import chunkSynapse


def create_groups(rankings: np.ndarray, group_size: int) -> tuple[list[np.ndarray[int]], list[range], list[np.ndarray[float]]]:
    """
    Creates groups of miners based on the rankings. The group size increases as the ranks get worse (higher number).
    There is always overlap between each group, with the size of the overlap being group_size // 2.

    Ex (assuming uids have ranks that match their uid):
    group_size = 2
    rankings = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    miner_groups = [[0, 1], [2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    group_ranks = [range(0, 2), range(1, 5), range(3, 9)]

    Args:
        rankings (np.ndarray): Array of rankings for the miners.
        group_size (int): Minimum number of miners in each group.

    Returns:
        tuple: A tuple containing:
            - miner_groups (list[np.array]): List of arrays of uids for each miner group.
            - group_ranks (list[range]): List of ranges for each miner group.
    """
    group_ranks = []
    miner_groups = []

    start = 0
    step = group_size // 2
    i = 0

    # create group of miners and ranks, increasing the group size by step each time
    while start < len(rankings) - step * 2:

        ranks_in_group = range(start, start + step * 2)
        # bt.logging.debug(f"ranks_in_group: {ranks_in_group}")
        group_ranks.append(ranks_in_group)
        miner_groups.append(np.array(rankings[ranks_in_group], dtype=int))

        # bt.logging.debug(
        #     f"start: {start}, step: {step}, added ranks: {group_ranks[-1]}, miners: {miner_groups[-1]}"
        # )

        start += step
        step += 1

    # if there are any miners left, add them to a group, handling edge case where no groups were created
    if start < len(rankings):
        if len(group_ranks) > 0:
            group_ranks[-1] = range(list(group_ranks[-1])[0], len(rankings))
        else:
            group_ranks.append(range(0, len(rankings)))
        if len(miner_groups) > 0:
            miner_groups[-1] = np.array(rankings[group_ranks[-1]], dtype=int)
        else:
            miner_groups.append(np.array(rankings[group_ranks[-1]], dtype=int))

    # bt.logging.debug(f"group_ranks: {group_ranks}")
    # bt.logging.debug(f"miner_groups: {miner_groups}")

    group_rank_values = []

    for i in range(len(miner_groups)):
        # bt.logging.debug(f"i: {i}, group_rank_values: {group_rank_values}")
        if i == 0:
            rank_values_for_group = [0, 1]
        elif i == 1:
            rank_values_for_group = [0.5, 1.5, 2.5, 3.5]
        else:
            second_most_recent_group_rank_values = group_rank_values[-2]
            last_rank_of_second_most_recent_group = (
                second_most_recent_group_rank_values[-1]
            )
            last_group_rank_values = group_rank_values[-1]
            overlap_index = len(last_group_rank_values) - i
            last_group_overlap_rank_value = last_group_rank_values[overlap_index]

            group_size = len(miner_groups[i])
            # bt.logging.debug(f"group_size: {group_size}")

            rank_start = (
                last_group_overlap_rank_value + last_rank_of_second_most_recent_group
            ) / 2
            rank_values_for_group = []
            for i in range(group_size):
                rank_values_for_group.append(rank_start + i)

        group_rank_values.append(np.array(rank_values_for_group, dtype=np.float64))

    return (miner_groups, group_ranks, group_rank_values)


def get_miner_groups(
    self,
) -> tuple[list[np.ndarray[int]], list[range], list[np.ndarray[float]]]:
    bt.logging.debug(f"rankings: {self.rankings}, sample_size: {self.sample_size}")
    group_size = min(len(self.rankings), self.sample_size)
    bt.logging.debug(f"group_size {group_size}")

    return create_groups(self.rankings, group_size)


def get_miner_groups_to_query(
    miner_groups: list[np.ndarray[np.int32]],
    num_miner_groups_to_query: int,
    choose_miner_group_index: int | None = None,
) -> list[int]:

    if choose_miner_group_index is not None:
        assert choose_miner_group_index >= 0 and choose_miner_group_index < len(
            miner_groups
        ), f"choose_miner_group_index out of bounds: index {choose_miner_group_index} not in range(0, {len(miner_groups)})"
        return [choose_miner_group_index]
    # else return random group
    miner_group_indices = random.sample(
        range(len(miner_groups)), num_miner_groups_to_query
    )
    return miner_group_indices


def get_alpha(
    self,
    num_miner_groups: int,
    miner_group_index: int,
    override_min_moving_average_alpha: float | None = None,
):
    """
    "tiered" alpha, where the alpha is higher for miner groups that have a lower rank (higher number)
    Ex:
        the first miner group as the "highest rank" and therefore the lowest alpha value
        the last miner group as the "lowest rank" and therefore the highest alpha

    This means that the scores are updated more slowly at higher ranks, because these miners should only be punished
    if they _consistently_ produce low quality responses. At lower ranks, the alpha value is higher, this allows for
    higher variability in the scores at lower ranks, allowing new miners with high quality responses to rise the ranks
    more quickly.

    Args:
        num_miner_groups (int): The number of miner groups.
        miner_group_index (int): The index of the miner group.
        override_min_moving_average_alpha (float | None): The alpha to use if the override is provided.

    Returns:
        float: The alpha value.
    """
    min_moving_average_alpha = (
        override_min_moving_average_alpha
        if override_min_moving_average_alpha
        else self.config.neuron.min_moving_average_alpha
    )
    alpha_adjustment = (1 - min_moving_average_alpha) / max((num_miner_groups - 1), 1)
    alpha = min_moving_average_alpha + alpha_adjustment * miner_group_index

    return alpha


async def query_miner_group(
    self,
    input_synapse: chunkSynapse,
    miner_group_uids: np.ndarray[np.int32],
    miner_group_index: int | None = None,
) -> list[chunkSynapse]:
    bt.logging.debug(
        f"Querying miner group ({miner_group_index}): {miner_group_uids}, timeout: {input_synapse.timeout}"
    )
    axons: list[bt.axon] = [self.metagraph.axons[uid] for uid in miner_group_uids]

    responses: list[chunkSynapse] = await self.query_axons(
        axons=axons,
        synapse=input_synapse,
        timeout=input_synapse.timeout,
    )

    bt.logging.debug(
        f"Got {len(responses)} responses from miner group ({miner_group_index}): {miner_group_uids}"
    )
    return responses


async def query_miner_groups(
    self,
    input_synapse: chunkSynapse,
    num_miner_groups_to_query: int = 1,
    choose_miner_group_index: int | None = None,
    custom_miner_uids: list[int] | None = None,
) -> tuple[
    list[list[chunkSynapse]],
    list[int | None],
    list[list[int]],
    list[np.ndarray[np.float64 | None]],
]:
    if custom_miner_uids is None:
        miner_groups, _, group_rank_values = get_miner_groups(self)

        miner_group_indices = get_miner_groups_to_query(
            miner_groups,
            num_miner_groups_to_query,
            choose_miner_group_index,
        )

        miner_uids_per_group = [miner_groups[i] for i in miner_group_indices]
        rank_values_per_group = [group_rank_values[i] for i in miner_group_indices]
    else:
        # custom group
        miner_group_indices = [None]
        miner_uids_per_group = [custom_miner_uids]
        rank_values_per_group = np.array([[None] * len(custom_miner_uids)])

    bt.logging.debug(f"Miner group indices: {miner_group_indices}")

    coros = []
    for miner_group_index, miner_group_uids in zip(
        miner_group_indices, miner_uids_per_group
    ):
        coros.append(
            query_miner_group(
                self,
                input_synapse,
                miner_group_uids=miner_group_uids,
                miner_group_index=miner_group_index,
            )
        )

    group_responses = await asyncio.gather(*coros)

    return (
        group_responses,
        miner_group_indices,
        miner_uids_per_group,
        rank_values_per_group,
    )


async def score_miner_group_responses(
    self,
    task: Task,
    responses: list[chunkSynapse],
    miner_group_uids: np.ndarray[np.int32],
    group_rank_values: np.ndarray[np.float64 | None],
    miner_group_index: int | None,
    do_wandb_log: bool,
    request_type: ChunkRequestType,
    reward_options: RewardOptions,
) -> EndTournamentRoundInfo | None:
    """
    Calculating rewards + ranking, making wandb data, making tournament round info for use in update_scores()
    """
    try:
        input_synapse = task.synapse

        bt.logging.debug("calling get_rewards() async")
        rewards, extra_infos = await get_rewards(
            document=input_synapse.document,
            chunk_size=input_synapse.chunk_size,
            chunk_qty=input_synapse.chunk_qty,
            responses=responses,
            client=AsyncOpenAI(),
            num_embeddings=self.num_embeddings,
            reward_options=reward_options,
            verbose=self.is_debug,
        )

        print(
            f"Rewards for {task.task_type} tournament round, Doc length: {len(input_synapse.document)}, Group index:"
        )
        pretty_print_rewards(miner_group_uids, rewards, extra_infos)

        ranked_responses = rank_responses(rewards)

        bt.logging.debug(f"Ranked responses: {ranked_responses}")

        any_none_group_rank_values = any(
            group_rank_value is None for group_rank_value in group_rank_values
        )

        if any_none_group_rank_values:
            # signifies there are no meaningful global ranks for custom group
            ranked_responses_global = np.array([-2] * len(ranked_responses)).astype(
                np.float64
            )
        else:
            # get rank values, "effective" rank that should be used when updating scores
            ranked_responses_global = rank_responses_global(
                self, group_rank_values, ranked_responses, miner_group_uids
            )

        bt.logging.debug(f"Rank values: {ranked_responses_global}")

        if miner_group_index is None:
            alpha = -1
        else:
            alpha = get_alpha(self, len(miner_group_uids), miner_group_index)

        scores: np.ndarray[np.float64] = self.scores
        rankings: np.ndarray[np.int32] = self.rankings

        wandb_data = make_wandb_data(
            block_number=self.block,
            miner_group_uids=miner_group_uids.astype(int).tolist(),
            miner_group_index=miner_group_index or -1,
            task=task,
            responses=responses,
            rewards=rewards,
            reward_extra_infos=extra_infos,
            ranked_responses=ranked_responses.astype(int).tolist(),
            ranked_responses_global=ranked_responses_global.astype(float).tolist(),
            alpha=alpha,
            request_type=request_type,
            is_debug=self.is_debug,
            cur_scores=scores.tolist(),
            cur_rankings=rankings.tolist(),
        )

        end_tournament_round_info = EndTournamentRoundInfo(
            responses=responses,
            miner_group_index=miner_group_index or -1,
            rewards=rewards.tolist(),
            rank_values=ranked_responses_global.tolist(),
            miner_group_uids=miner_group_uids.astype(int).tolist(),
            alpha=alpha,
            do_wandb_log=do_wandb_log,
            wandb_data=wandb_data,
            task_type=task.task_type,
            group_best_possible_rank_value=group_rank_values[0]
        )

        # bt.logging.debug(f"End tournament round info: {end_tournament_round_info}")
        return end_tournament_round_info

    except Exception as e:
        bt.logging.error(f"Error querying miner group: {e}")
        bt.logging.error(traceback.format_exc())
        return None


async def run_tournament_round(
    self,
    task: Task,
    do_wandb_log: bool,
    choose_miner_group_index: int | None = None,
    custom_miner_uids: (
        list[int] | None
    ) = None,  # takes precedence over `choose_miner_group_index`
    request_type: ChunkRequestType = ChunkRequestType.normal,
    reward_options: RewardOptions = RewardOptions(),
    # TODO: do not score responses if the task is not do_scoring
) -> list[EndTournamentRoundInfo | None]:
    """
    Run a tournament round for the validator's tournament.
    """

    (
        group_responses,
        miner_group_indices,
        miner_uids_per_group,
        rank_values_per_group,
    ) = await query_miner_groups(
        self,
        task.synapse,
        num_miner_groups_to_query=1,
        choose_miner_group_index=choose_miner_group_index,
        custom_miner_uids=custom_miner_uids,
    )

    bt.logging.debug(
        f"Got {len(group_responses)} group responses from groups: {miner_uids_per_group}"
    )

    coros = []
    for miner_group_responses, miner_group_index, miner_uids, rank_values in zip(
        group_responses,
        miner_group_indices,
        miner_uids_per_group,
        rank_values_per_group,
    ):
        bt.logging.debug(
            f"Scoring {len(miner_group_responses)} responses from group {miner_uids}"
        )
        coros.append(
            score_miner_group_responses(
                self,
                task=task,
                responses=miner_group_responses,
                miner_group_uids=np.array(miner_uids),
                group_rank_values=rank_values,
                miner_group_index=miner_group_index,
                do_wandb_log=do_wandb_log,
                request_type=request_type,
                reward_options=reward_options,
            )
        )

    return await asyncio.gather(*coros)
