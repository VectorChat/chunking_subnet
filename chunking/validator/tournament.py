import asyncio
from typing import Optional
import traceback
import bittensor as bt
import numpy as np
from random import choice
from chunking.validator.reward import get_rewards, rank_responses, rank_responses_global
from chunking.validator.types import EndTournamentRoundInfo
from chunking.protocol import chunkSynapse


def create_groups(rankings: np.ndarray, group_size: int):
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

    bt.logging.debug(f"group_ranks: {group_ranks}")
    bt.logging.debug(f"miner_groups: {miner_groups}")

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
    choose_miner_uid: int | None = None,
    choose_miner_group_index: int | None = None,
) -> list[int]:

    if choose_miner_group_index is not None:
        assert choose_miner_group_index >= 0 and choose_miner_group_index < len(
            miner_groups
        ), f"choose_miner_group_index out of bounds: index {choose_miner_group_index} not in range(0, {len(miner_groups)})"
        return [choose_miner_group_index]
    elif choose_miner_uid is not None:
        groups = set()
        for group_index in range(len(miner_groups)):
            if choose_miner_uid in miner_groups[group_index]:
                groups.add(group_index)
                break

        return list(groups)

    # else return random group
    miner_group_indices = choice(range(len(miner_groups)), num_miner_groups_to_query)
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
    miner_group_index: int,
) -> list[chunkSynapse]:
    bt.logging.debug(
        f"Querying miner group ({miner_group_index}): {miner_group_uids}, timeout: {input_synapse.timeout}"
    )
    axons: list[bt.axon] = [self.metagraph.axons[uid] for uid in miner_group_uids]
    responses: list[chunkSynapse] = self.dendrite.query(
        axons=axons,
        timeout=input_synapse.timeout,
        synapse=input_synapse,
        deserialize=False,
    )
    return responses


async def query_miner_groups(
    self,
    input_synapse: chunkSynapse,
    num_miner_groups_to_query: int = 1,
    choose_miner_index: int | None = None,
    choose_miner_group_index: int | None = None,
    return_group_indices: bool = False,
) -> list[list[chunkSynapse]] | list[tuple[list[chunkSynapse], list[int]]]:
    miner_groups, _, _ = get_miner_groups(self)

    choose_miner_uid = int(self.rankings[choose_miner_index]) if choose_miner_index else None

    miner_group_indices = get_miner_groups_to_query(
        miner_groups,
        num_miner_groups_to_query,
        choose_miner_uid,
        choose_miner_group_index,
    )

    miner_groups_to_query = [miner_groups[i] for i in miner_group_indices]

    coros = []
    for i, (miner_group_uids) in enumerate(miner_groups_to_query):
        coros.append(
            query_miner_group(
                self,
                input_synapse,
                miner_group_uids=miner_group_uids,
                miner_group_index=i,
            )
        )

    group_responses = await asyncio.gather(*coros)

    if return_group_indices:
        return group_responses, miner_group_indices
    else:
        return group_responses


async def score_miner_group_responses(
    self,
    input_synapse: chunkSynapse,
    responses: list[chunkSynapse],
    miner_group_uids: np.ndarray[np.int32],
    group_rank_values: np.ndarray[np.float64],
    miner_group_index: int,
) -> EndTournamentRoundInfo | None:
    try:
        rewards, extra_infos = get_rewards(
            self,
            document=input_synapse.document,
            chunk_size=input_synapse.chunk_size,
            chunk_qty=input_synapse.chunk_qty,
            responses=responses,
        )

        bt.logging.debug(f"Rewards: {rewards}")

        ranked_responses = rank_responses(rewards)

        bt.logging.debug(f"Ranked responses: {ranked_responses}")

        ranked_responses_global = rank_responses_global(
            self, group_rank_values, ranked_responses, miner_group_uids
        )

        bt.logging.debug(f"Ranked responses global: {ranked_responses_global}")

        return EndTournamentRoundInfo(
            responses=responses,
            miner_group_index=miner_group_index,
            rewards=rewards.tolist(),
            ranked_responses_global=ranked_responses_global.tolist(),
            miner_group_uids=miner_group_uids.tolist(),
            task_type="organic",
            alpha=get_alpha(self, len(miner_group_uids), miner_group_index),
        )
    except Exception as e:
        bt.logging.error(f"Error querying miner group: {e}")
        bt.logging.error(traceback.format_exc())
        return None


async def run_tournament_round(
    self,
    input_synapse: chunkSynapse,
    choose_miner_index: int | None = None,
    choose_miner_group_index: int | None = None,
) -> list[EndTournamentRoundInfo | None]:
    """
    Run a tournament round for the validator's tournament.
    """

    group_responses, miner_group_indices = await query_miner_groups(
        self,
        input_synapse,
        num_miner_groups_to_query=1,
        choose_miner_index=choose_miner_index,
        choose_miner_group_index=choose_miner_group_index,
        return_group_indices=True,
    )

    miner_groups, _, group_rank_values = get_miner_groups(self)

    coros = []
    for miner_group_responses, miner_group_index in zip(
        group_responses, miner_group_indices
    ):
        coros.append(
            score_miner_group_responses(
                self,
                input_synapse,
                responses=miner_group_responses,
                miner_group_uids=miner_groups[miner_group_index],
                group_rank_values=group_rank_values[miner_group_index],
                miner_group_index=miner_group_index,
            )
        )

    return await asyncio.gather(*coros)
