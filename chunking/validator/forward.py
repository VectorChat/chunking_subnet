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

import bittensor as bt
from random import choice
from math import floor
import numpy as np
from chunking.protocol import chunkSynapse
from chunking.validator.reward import get_rewards, rank_responses
from chunking.validator.task_api import Task
from neurons.validator import Validator
import json
import gzip
import base64

def get_miner_groups(self: Validator) -> tuple[np.ndarray, np.ndarray, int]:
    bt.logging.debug(f"rankings: {self.rankings}, sample_size: {self.sample_size}")
    group_size = min(len(self.rankings), self.sample_size)    
    bt.logging.debug(f"group_size {group_size}")    
    group_ranks = []
    miner_groups: list[np.array] = []

    start = 0
    stop = len(self.rankings) - group_size + 1
    step = floor(group_size / 2)
    
    bt.logging.debug(f"start: {start}, stop: {stop}, step: {step}")
    
    for i in range(start, stop, step):
        group_ranks.append(range(i, i+group_size))
        miner_groups.append(np.array(self.rankings[group_ranks[-1]], dtype=int))
    return (miner_groups, group_ranks, group_size)
    
    
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
    miner_groups, group_ranks, group_size = get_miner_groups(self)  
    
    bt.logging.debug(f"Miner groups: {miner_groups}")
    bt.logging.debug(f"Group ranks: {group_ranks}")
    bt.logging.debug(f"Group size: {group_size}")    
      
    task = Task.get_new_task(validator=self)

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
    
    miner_group_uids = list(map(lambda x: int(x), miner_groups[miner_group]))  
    
    wandb_data["group"]["uids"] = miner_group_uids  
    
    bt.logging.debug(f"Quering miner group: {miner_group}, with uids: {miner_group_uids}, timeout: {task.synapse.timeout}")
    
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

    for response, uid in zip(responses, miner_group_uids):
        if response.dendrite.process_time is None:
            wandb_data["group"]["process_times"][str(uid)] = np.inf
        else:
            wandb_data["group"]["process_times"][str(uid)] = response.dendrite.process_time
        
        # wandb_data["group"]["num_chunks"][str(uid)] = len(response.chunks) if response.chunks is not None else 0
        
        if response.chunks is None:            
            continue
        
        json_str = json.dumps(response.chunks)
        compressed = gzip.compress(json_str.encode())
        encoded = base64.b64encode(compressed).decode()
        
        wandb_data["group"]["chunks"][str(uid)] = encoded

    def print_response(response: chunkSynapse):                
        num_chunks = len(response.chunks) if response.chunks is not None else 0
        sig = response.miner_signature[:10] + "..." if response.miner_signature is not None else "No signature found"
        
        string = f"{response.axon.hotkey[:10]}: received {num_chunks} chunks"        
        string += f", signature: {sig}, total_size: {response.total_size} bytes"
        
        bt.logging.debug(string)
    
    bt.logging.debug("Responses:")
    for response in responses:        
        print_response(response)    
    
    hotkeys = [response.axon.hotkey for response in responses]
    
    for uid in miner_group_uids:
        if uid in hotkeys:
            wandb_data["group"]["process_times"][str(uid)] = np.inf
        else:
            wandb_data["group"]["process_times"][str(uid)] = 0
    
    rewards = get_rewards(
        self,
        document=task.synapse.document,
        chunk_size=task.synapse.chunk_size,
        responses=responses,
    )
    
    for reward, uid in zip(rewards, miner_group_uids):
        wandb_data["group"]["rewards"][str(uid)] = reward

    bt.logging.debug(f"Scored responses: {rewards}")
    
    log_data = {
        'hotkey': hotkey.ss58_address,
        'nonce': self.step,
        'task_id': task.task_id,
        'miner_uids': miner_group_uids,
        'rewards': rewards.tolist(),
    }
    
    bt.logging.debug(f"log_data: {log_data}")

    # Task.upload_logs(self, log_data)
    
    ranked_responses = rank_responses(rewards)    
    
    for rank, uid in zip(ranked_responses, miner_group_uids):
        wandb_data["group"]["local_rankings"][str(uid)] = rank
    
    # inf means the response should not be ranked
    ranked_responses_global = np.full_like(ranked_responses, np.inf)
    
    # Offset ranks by the group offset to get 'global' ranks
    group_offset = group_ranks[miner_group][0]

    for i, rank in enumerate(ranked_responses):
        if rank != -1:
            ranked_responses_global[i] = ranked_responses[i] + group_offset                      
        elif not np.isinf(self.scores[miner_group_uids[i]]):
            # give response worst rank in the group
            ranked_responses_global[i] = (
                group_offset +
                np.sum(np.abs(ranked_responses))
            )       

    for rank, uid in zip(ranked_responses_global, miner_group_uids):
        wandb_data["group"]["global_rankings"][str(uid)] = rank

    bt.logging.info(f"Ranked responses: {ranked_responses}")
    bt.logging.info(f"Global ranked responses: {ranked_responses_global}")

    if task.task_type == "organic":
        if task.miner_uids is None or not found_match:
            response = responses[ranked_responses.argmin()]
            
        else:
            for i in range(len(task.miner_uids)):
                if task.miner_uids[i] in miner_group_uids:
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
    self.update_scores(wandb_data, ranked_responses_global, miner_groups[miner_group], task.task_type)
    # time.sleep(5)
