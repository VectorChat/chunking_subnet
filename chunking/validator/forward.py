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
import bittensor as bt
from random import choice
from math import ceil
import numpy as np
from chunking.validator.reward import get_rewards, rank_responses
from chunking.utils.uids import get_random_uids
from chunking.validator.task_api import Task

def get_miner_groups(self) -> (np.ndarray, np.ndarray, int):
    group_size = min(len(self.rankings), self.sample_size)
    group_ranks = []
    miner_groups = []
    for i in range(-floor(group_size / 2), len(self.rankings) - group_size + 1, floor(group_size / 2)):
        group_ranks.append(range(i, i+group_size))
        miner_groups.append(np.array(self.rankings[group_ranks[-1]]))
    return (miner_groups, group_ranks, group_size)
    

async def forward(self):
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


    hotkey = self.wallet.get_hotkey()
    miner_groups, group_ranks, group_size = get_miner_groups(self)    
    task = Task.get_new_task(self)

    if task.miner_uids is not None:
        found_match = False
        for uid in task.miner_uids:
            if found_match:
                break
            for i in range(1, len(miner_groups)):
                if uid in miner_groups[i]:
                    miner_group = i
                    found_match = True
                    break
                
    if task.miner_uids is None or not found_match:
        miner_group = choice(range(len(miner_groups)))
    
    # The dendrite client queries the network.
    responses = self.dendrite.query(
        axons=[self.metagraph.axons[uid] for uid in miner_groups[miner_group]],
        timeout=task.synapse.timeout,
        synapse=task.synapse,
        deserialize=False,
    )

    bt.logging.info(f"Received responses: {responses}")
    rewards = get_rewards(self, document=task.synapse.document, chunk_size=task.synapse.chunk_size, responses=responses)
    bt.logging.debug(f"Scored responses: {rewards}")
    
    log_data = {
        'hotkey': hotkey.ss58_address,
        'nonce': self.step,
        'task_id': task.task_id,
        'miner_uids': miner_groups[miner_group].tolist(),
        'rewards': rewards.tolist(),
    }

    Task.upload_logs(self, log_data)

    if miner_group != 0:
        response_ranks = np.array(
            [reward + group_ranks[miner_group][0] for reward in rank_responses(rewards)]
            )
    else:
        response_ranks = np.concatenate((
            [
                reward + len(self.rankings) + group_ranks[miner_group][0]
                for reward in rank_responses(rewards[:(group_size // 2)])
            ], 
            [
                reward for reward in rank_responses(rewards[(group_size // 2):])
            ]
        ))

    bt.logging.info(f"Ranked responses: {response_ranks}")

    if task.task_type == "organic":
        if task.miner_uids is None or not found_match:
            response = responses[response_ranks.argmin()]
            
        else:
            for i in range(len(task.miner_uids)):
                if task.miner_uids[i] in miner_groups[miner_group]:
                    response = responses[i]
                    break

        task_data = {
            'document': response.document,
            'chunk_size': response.chunk_size,
            'chunks': response.chunks,
        }

        response_data = {
            'task_data': task_data,
            'miner_signature': response.miner_signature,
            'miner_hotkey': response.axon.hotkey,
            'validator_hotkey': hotkey.ss58_address,
            'task_id': task.task_id,            
            'nonce': self.step,
        }

        Task.return_response(self, response_data)
    self.update_scores(response_ranks, miner_groups[miner_group])
    time.sleep(5)
