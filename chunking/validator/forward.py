# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 VectorChat

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import traceback
import bittensor as bt
from random import choice
from math import floor
import numpy as np
from chunking.protocol import chunkSynapse
from chunking.utils import uids
from chunking.validator.reward import get_rewards, rank_responses, rank_responses_global
from chunking.validator.task_api import Task
from neurons.validator import Validator
import json
import gzip
import base64
from tabulate import tabulate


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
    self: Validator,
) -> tuple[list[np.ndarray[int]], list[range], list[np.ndarray[float]]]:
    bt.logging.debug(f"rankings: {self.rankings}, sample_size: {self.sample_size}")
    group_size = min(len(self.rankings), self.sample_size)
    bt.logging.debug(f"group_size {group_size}")

    return create_groups(self.rankings, group_size)


def get_alpha(
    self: Validator,
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


async def forward(self: Validator):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Handles 2 cases:
     - Organic query coming in through API
     - Generated query when there are no queries coming in

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.
        synapse: The chunkSynapse containing the organic query
    """
    # initial structure for wandb logging
    wandb_data = {
        "modality": "text",
        "sync_block": self.block,
        "all": {
            "scores": {},
            "rankings": {},
        },
        "group": {
            "process_times": {},
            "rewards": {},
            "local_rankings": {},
            "global_rankings": {},
            "scores": {},
            "chunks": {},
        },
    }

    hotkey = self.wallet.get_hotkey()

    # get miner groups and with their ranks for the tournament round
    miner_groups, group_ranks, group_rank_values = get_miner_groups(self)

    # bt.logging.debug(f"Miner groups: {miner_groups}")
    # bt.logging.debug(f"Group ranks: {group_ranks}")
    # bt.logging.debug(f"Group rank values: {group_rank_values}")

    # get new task to query miners with
    # this gets either an organic query from the API or a synthetic query (currently wikipedia)
    try:
        task = await Task.get_new_task(validator=self)
    except Exception as e:
        bt.logging.error(f"Error getting new task: {e}")
        bt.logging.error(traceback.format_exc())
        return

    # log pageid of wikipedia article used for synthetic query
    wandb_data["pageid"] = task.page_id or -1

    # if there are uids to query (organic query), choose a random miner group from the tournament round that contains at least one uid from the organic query
    # if there are no uids to query (synthetic query), choose a random miner group from the tournament round
    if task.miner_uids is not None:
        # choose least number of groups that contain all the uids

        groups = set()

        for uid in task.miner_uids:
            for group_index in range(len(miner_groups)):
                if uid in miner_groups[group_index]:
                    groups.add(group_index)

        miner_group = choice(list(groups))
    else:
        miner_group = choice(range(len(miner_groups)))

    miner_group_uids = list(map(lambda x: int(x), miner_groups[miner_group]))

    wandb_data["group"]["uids"] = miner_group_uids

    # log the uids that are part of the miner group that is queried
    wandb_data["group"]["miner_group"] = miner_group

    bt.logging.debug(
        f"Quering miner group: {miner_group}, with uids: {miner_group_uids}, timeout: {task.synapse.timeout}"
    )

    bt.logging.debug(f"group ranks: {group_ranks[miner_group]}")
    bt.logging.debug(f"group rank values: {group_rank_values[miner_group]}")

    # get the axons (miners) that are part of the miner group that is queried
    axons: list[bt.axon] = [self.metagraph.axons[uid] for uid in miner_group_uids]

    bt.logging.debug(f"Querying axons: {axons}")

    # The dendrite client queries the network.
    try:
        responses: list[chunkSynapse] = self.dendrite.query(
            axons=axons,
            timeout=task.synapse.timeout,
            synapse=task.synapse,
            deserialize=False,
        )
    except Exception as e:
        bt.logging.error(f"Error querying the network: {e}")

    # list to store the process times for each response
    process_times = []

    # loop through responses and get process times and chunks for each response
    for response, uid in zip(responses, miner_group_uids):
        if response.dendrite.process_time is None:
            wandb_data["group"]["process_times"][str(uid)] = np.inf
            process_times.append(np.inf)
        else:
            wandb_data["group"]["process_times"][
                str(uid)
            ] = response.dendrite.process_time
            process_times.append(response.dendrite.process_time)

        # wandb_data["group"]["num_chunks"][str(uid)] = len(response.chunks) if response.chunks is not None else 0

        if response.chunks is None:
            continue

        # compress and encode the chunks for wandb logging
        json_str = json.dumps(response.chunks)
        compressed = gzip.compress(json_str.encode())
        encoded = base64.b64encode(compressed).decode()

        wandb_data["group"]["chunks"][str(uid)] = encoded

    # function to print a miner's response, useful for debugging
    def print_response(response: chunkSynapse):
        num_chunks = len(response.chunks) if response.chunks is not None else 0
        sig = (
            response.miner_signature[:10] + "..."
            if response.miner_signature is not None
            else "No signature found"
        )

        string = f"{response.axon.hotkey[:10]}: received {num_chunks} chunks"
        string += f", signature: {sig}, total_size: {response.total_size} bytes"

        bt.logging.debug(string)

    bt.logging.debug("Responses:")
    for response in responses:
        print_response(response)

    # list of hotkeys from the responses
    hotkeys = [response.axon.hotkey for response in responses]

    # log the hotkeys from the responses for wandb logging
    wandb_data["group"]["hotkeys"] = hotkeys

    # get the rewards for each response
    rewards, extra_infos = get_rewards(
        self,
        document=task.synapse.document,
        chunk_size=task.synapse.chunk_size,
        chunk_qty=task.synapse.chunk_qty,
        responses=responses,
    )

    # log the rewards for each response for wandb logging

    for reward, uid in zip(rewards, miner_group_uids):
        wandb_data["group"]["rewards"][str(uid)] = reward

    # print a neat table to show rewards and other metrics for each miner response
    table_data = []
    for i, tuple_info in enumerate(zip(miner_group_uids, rewards, extra_infos)):
        uid, reward, extra_info = tuple_info
        embedding_reward = extra_info.get("embedding_reward", "n/a")
        size_penalty = extra_info.get("size_penalty", "n/a")
        qty_penalty = extra_info.get("qty_penalty", "n/a")
        time_penalty = extra_info.get("time_penalty", "n/a")
        num_embed_tokens = extra_info.get("num_embed_tokens", "n/a")

        table_data.append(
            (
                uid,
                reward,
                embedding_reward,
                size_penalty,
                qty_penalty,
                time_penalty,
                num_embed_tokens,
            )
        )

    # sort the table data by reward in descending order
    sorted_desc = sorted(table_data, key=lambda x: x[1], reverse=True)

    print("\nRewards and UIDs:")
    print(
        tabulate(
            sorted_desc,
            headers=[
                "UID",
                "Reward",
                "Embedding Reward",
                "Size Penalty",
                "Quantity Penalty",
                "Time Penalty",
                "Num Embed Tokens",
            ],
            tablefmt="grid",
        )
    )

    bt.logging.debug(f"Scored responses: {rewards}")

    # log data for task api logging not currently used
    log_data = {
        "hotkey": hotkey.ss58_address,
        "nonce": self.step,
        "task_id": task.task_id,
        "miner_uids": miner_group_uids,
        "rewards": rewards.tolist(),
    }

    # bt.logging.debug(f"log_data: {log_data}")

    # Task.upload_logs(self, log_data)

    # rank the miner responses based on the rewards within this miner group
    ranked_responses = rank_responses(rewards)

    # log the local rankings for each response for wandb logging
    for rank, uid in zip(ranked_responses, miner_group_uids):
        wandb_data["group"]["local_rankings"][str(uid)] = rank

    group_rank_values = group_rank_values[miner_group]

    ranked_responses_global = rank_responses_global(
        self,
        group_rank_values,
        ranked_responses,
        miner_group_uids,
    )

    # log the global rankings (rankings in global context of miners within this group) for each response for wandb logging
    for rank, uid in zip(ranked_responses_global, miner_group_uids):
        wandb_data["group"]["global_rankings"][str(uid)] = rank

    bt.logging.info(f"Ranked responses: {ranked_responses}")
    bt.logging.info(f"Global ranked responses: {ranked_responses_global}")

    # handle returning the organic query response with the highest reward to the task api
    if task.task_type == "organic":
        try:
            # get the response with highest reward
            index = np.argmax(rewards)

            bt.logging.info(
                f"Choosing response with index: {index}, reward: {rewards[index]}, rank: {ranked_responses[index]}"
            )

            response = responses[index]

            if isinstance(response.chunks, np.ndarray):
                chunks = response.chunks.tolist()
            elif response.chunks is None:
                chunks = []
            else:
                chunks = response.chunks

            task_data = {
                "document": response.document,
                "chunk_size": response.chunk_size,
                "chunk_qty": response.chunk_qty,
                "chunks": chunks,
            }

            response_data = {
                "task_data": task_data,
                "miner_signature": response.miner_signature,
                "miner_hotkey": response.axon.hotkey,
                "validator_hotkey": hotkey.ss58_address,
                "task_id": task.task_id,
                "nonce": time.time_ns(),
            }

            Task.return_response(self, response_data)
        except Exception as e:
            bt.logging.error(f"Error returning organic query response: {e}")

    # get the alpha for the miner group
    alpha = get_alpha(self, len(miner_groups), miner_group)

    # update the scores (moving average of rankings) for each miner, and set the new global ranking for all miners
    self.update_scores(
        wandb_data, ranked_responses_global, miner_group_uids, task.task_type, alpha
    )
    # time.sleep(5)
