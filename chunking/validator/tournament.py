import asyncio
from typing import Optional
import traceback
import bittensor as bt
import numpy as np
from random import choice
from chunking.validator.reward import get_rewards, rank_responses
from chunking.validator.types import EndTournamentRoundInfo
from chunking.protocol import chunkSynapse


def create_groups(
    rankings: np.ndarray[np.int32], group_size: int
) -> tuple[list[np.ndarray[np.int32]], list[range], int]:
    """
    Creates groups of miners based on the rankings. The group size increases as the ranks get worse (higher number).
    There is always overlap between each group, with the size of the overlap being group_size // 2.

    Ex (assuming uids have ranks that match their uid):
    group_size = 2
    rankings = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    miner_groups = [[0, 1], [2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    group_ranks = [range(0, 2), range(2, 6), range(6, 12)]

    Args:
        rankings (np.ndarray): Array of rankings for the miners.
        group_size (int): Minimum number of miners in each group.

    Returns:
        tuple: A tuple containing:
            - miner_groups (list[np.array]): List of arrays of uids for each miner group.
            - group_ranks (list[range]): List of ranges for each miner group.
    """
    group_ranks = []
    miner_groups: list[np.ndarray[np.int32]] = []

    start = 0
    step = group_size // 2

    # create group of miners and ranks, increasing the group size by step each time
    while start < len(rankings) - step * 2:
        group_ranks.append(range(start, start + step * 2))
        miner_groups.append(np.array(rankings[group_ranks[-1]], dtype=np.int32))
        start += step
        step += 1

    # if there are any miners left, add them to a group, handling edge case where no groups were created
    if start < len(rankings):
        if len(group_ranks) > 0:
            group_ranks[-1] = range(list(group_ranks[-1])[0], len(rankings))
        else:
            group_ranks.append(range(0, len(rankings)))
        if len(miner_groups) > 0:
            miner_groups[-1] = np.array(rankings[group_ranks[-1]], dtype=np.int32)
        else:
            miner_groups.append(np.array(rankings[group_ranks[-1]], dtype=np.int32))

    return (miner_groups, group_ranks, group_size)


def get_miner_groups(self) -> tuple[list[np.ndarray[np.int32]], list[range], int]:
    bt.logging.debug(f"rankings: {self.rankings}, sample_size: {self.sample_size}")
    group_size = min(len(self.rankings), self.sample_size)
    bt.logging.debug(f"group_size {group_size}")

    return create_groups(self.rankings, group_size)


def get_miner_groups_to_query(
    self,
    miner_groups: list[np.ndarray[np.int32]],
    group_ranks: list[range],
    filter_miner_uids: Optional[list[int]] = None,
) -> tuple[list[np.ndarray[np.int32]], list[range]]:

    if filter_miner_uids is not None:
        groups = set()
        for uid in filter_miner_uids:
            for group_index in range(len(miner_groups)):
                if uid in miner_groups[group_index]:
                    groups.add(group_index)
                    break

        miner_groups_to_query = [miner_groups[group] for group in groups]
        miner_groups_group_ranks = [group_ranks[group] for group in groups]
        return miner_groups_to_query, miner_groups_group_ranks

    # else return random group
    miner_group = choice(range(len(miner_groups)))
    return [miner_groups[miner_group]], [group_ranks[miner_group]]


def get_tiered_alpha(self, miner_group_index: int, miner_groups: list[np.ndarray]):
    """
    'tiered' alpha, where the alpha is higher for miner groups that have a lower rank (higher number)
    Ex:
       the first miner group as the "highest rank" and therefore the lowest alpha value
       the last miner group as the "lowest rank" and therefore the highest alpha value

    This means that the scores are updated more slowly at higher ranks, because these miners should only be punished
    if they _consistently_ produce low quality responses. At lower ranks, the alpha value is higher, this allows for
    higher variability in the scores at lower ranks, allowing new miners with high quality responses to rise the ranks
    more quickly.

    A
    """
    alpha_adjustment = (1 - self.config.neuron.min_moving_average_alpha) / (
        len(miner_groups) - 1
    )
    alpha = (
        self.config.neuron.min_moving_average_alpha
        + alpha_adjustment * miner_group_index
    )

    return alpha


async def query_miner_group(
    self,
    input_synapse: chunkSynapse,
    miner_group_uids: np.ndarray[np.int32],
    miner_group_ranks: range,
    miner_group_index: int,
    miner_groups: list[np.ndarray[np.int32]],
) -> EndTournamentRoundInfo | None:
    try:
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

        ranked_responses_global = get_ranked_responses_in_global_context(
            self, miner_group_uids, list(miner_group_ranks), ranked_responses
        )

        bt.logging.debug(f"Ranked responses global: {ranked_responses_global}")

        return EndTournamentRoundInfo(
            responses=responses,
            rewards=rewards.tolist(),
            ranked_responses_global=ranked_responses_global.tolist(),
            miner_group_uids=miner_group_uids.tolist(),
            task_type="organic",
            alpha=get_tiered_alpha(self, miner_group_index, miner_groups),
        )
    except Exception as e:
        bt.logging.error(f"Error querying miner group: {e}")
        bt.logging.error(traceback.format_exc())
        return None


async def run_tournament_round(
    self,
    input_synapse: chunkSynapse,
    miner_uids: Optional[list[int]] = None,
) -> list[EndTournamentRoundInfo | None]:
    """
    Run a tournament round for the validator's tournament.
    """

    miner_groups, group_ranks, group_size = get_miner_groups(self)

    miner_groups_to_query, miner_groups_group_ranks = get_miner_groups_to_query(
        self, miner_groups, group_ranks, miner_uids
    )

    coros = []
    for i, (miner_group, miner_group_rank) in enumerate(
        zip(miner_groups_to_query, miner_groups_group_ranks)
    ):
        coros.append(
            query_miner_group(
                self,
                input_synapse,
                miner_group,
                miner_group_rank,
                i,
                miner_groups,
            )
        )

    results = await asyncio.gather(*coros)

    return results


def get_ranked_responses_in_global_context(
    self,
    miner_group_uids: np.ndarray,
    miner_group_ranks: list[int],
    ranked_responses: np.ndarray,
):
    # inf means the response should not be ranked
    ranked_responses_global = np.full_like(ranked_responses, np.inf)

    # Offset ranks by the group offset to get 'global' ranks
    group_offset = miner_group_ranks[0]

    # loop through the ranked responses and assign a global rank to each response

    for i, rank in enumerate(ranked_responses):
        if rank != -1:
            ranked_responses_global[i] = ranked_responses[i] + group_offset
        elif not np.isinf(self.scores[miner_group_uids[i]]):

            absolutes = np.abs(ranked_responses)

            bt.logging.debug(f"absolutes: {absolutes}")

            sum_value = np.sum(absolutes)

            bt.logging.debug(f"sum_value: {sum_value}")
            # give response worst rank in the group
            ranked_responses_global[i] = group_offset + sum_value

    return ranked_responses_global
