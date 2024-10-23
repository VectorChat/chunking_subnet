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

import traceback
import bittensor as bt
from random import choice
import numpy as np
from chunking.protocol import chunkSynapse
from chunking.validator.reward import get_rewards, rank_responses
from chunking.validator.task_api import Task, TaskType
from chunking.validator.tournament import (
    get_miner_groups,
)
from chunking.utils import uids
from chunking.validator.reward import get_rewards, rank_responses, rank_responses_global
from chunking.validator.task_api import Task
from neurons.validator import Validator
import json
import gzip
import base64
from tabulate import tabulate
from chunking.validator.types import EndTournamentRoundInfo



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
    try:
        task = await Task.get_new_task(validator=self)
    except Exception as e:
        bt.logging.error(f"Error getting new synthetic task: {e}")
        bt.logging.error(traceback.format_exc())
        return

    # log pageid of wikipedia article used for synthetic query
    wandb_data["pageid"] = task.page_id or -1

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

    alpha = get_tiered_alpha(self, miner_group, miner_groups)

    end_tournament_round_info = EndTournamentRoundInfo(
        alpha=alpha,
        miner_group_uids=miner_group_uids,
        ranked_responses_global=ranked_responses_global,
        task_type=task.task_type,
        responses=responses,
        rewards=rewards,
        wandb_data=wandb_data,
    )

    await self.queue_score_update(end_tournament_round_info)
