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
from math import ceil
import numpy as np
from chunking.protocol import chunkSynapse
from chunking.validator.reward import rank_responses
from chunking.utils.uids import get_random_uids
import requests
import subprocess
import time
import os

async def forward(self, synapse: chunkSynapse=None):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Handles 3 cases:
     - Organic query coming in through the axon
     - Organic query coming in through API
     - Generated query when there are no queries coming in

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.
        synapse: The chunkSynapse containing the organic query
    """

    # don't grab miner ids if they are passed in from the organic axon request
    if self.sample_size > len(self.rankings):
        miner_uids = get_random_uids(self, k=min(self.sample_size, self.metagraph.n.item()))
        ranks = np.zeros_like(miner_uids)
        minerGroup = 1
    else:
        minerGroup = choice(range(ceil(2 * len(self.rankings) / self.sample_size)))
        centerRank = minerGroup * self.sample_size // 2
        if centerRank > len(self.rankings) - ceil(self.sample_size / 2):
            ranks = range((floor(centerRank) - (self.sample_size // 2)), len(self.rankings))
        else:
            ranks = range((floor(centerRank) - (self.sample_size // 2)), (floor(centerRank) + ceil(self.sample_size / 2)))
        miner_uids = np.array(self.rankings[ranks])
    bt.logging.info(f"Getting miner uids: {miner_uids}")


    if synapse:
        bt.logging.debug("Synapse was passed in")
        # if len(synapse.miner_uids) > 0:
        #     miner_uids = synapse.miner_uids
        #     bt.logging.info(f"Parsed uids: {miner_uids}")
        
        if not synapse.timeout:
            synapse.timeout = 10.0
        
        if not synapse.maxTokensPerChunk:
            synapse.maxTokensPerChunk = 200

    else:
        page = 312990#choice([312990, 9046237, 585013, 444081, 12559806, 30873232, 9236, 9577500, 21501970])
        # page = requests.get('https://en.wikipedia.org/w/api.php', params={
        #     'action': 'query',
        #     'format': 'json',
        #     'list': 'random',
        #     'rnnamespace': 0,
        # }).json()['query']['random'][0]['id']

        document_text = requests.get('https://en.wikipedia.org/w/api.php', params={
            'action': 'query',
            'format': 'json',
            'pageids': page,
            'prop': 'extracts',
            'explaintext': True,
            'exsectionformat': 'plain',
            }).json()['query']['pages'][str(page)]['extract']
        document_text = document_text.replace("\n", " ").replace("\t", " ")
        document_text = ' '.join(document_text.split())
        filename = str(hash(time.time())) + '.txt'
        with open(filename, 'x') as file:
            file.write(document_text)
            file.close()
        bt.logging.debug(f"Wrote document text to {filename}")
        cid = subprocess.run(["ipfs", "add", filename], capture_output=True).stdout[6:52]
        os.remove(filename)
        bt.logging.debug(f"Uploaded document with cid: {cid}")
        synapse = chunkSynapse(document=cid, timeout=5.0, maxTokensPerChunk=200)
    
    # The dendrite client queries the network.
    responses = self.dendrite.query(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=False,
        timeout=synapse.timeout,
    )
    subprocess.run(['ipfs', 'pin', 'rm', cid])
    bt.logging.info(
        "Received responses: ", 
        ['\n' + f'{miner_uids[i]} ({ranks[i]}): {[chunk[:10] + ' ...' for chunk in responses[i].chunks]}' for i in range(len(responses))]
        ) 
    if minerGroup != 0:
        responseRanks = [reward + ranks[0] for reward in rank_responses(self, document=document_text, responses=responses)]
    else:
        responseRanks = np.concatenate([reward + ranks[0] + len(self.rankings) for reward in rank_responses(self, document=document_text, responses=responses[:(self.sample_size // 2)])], 
            [reward for reward in rank_responses(self, document=document_text, responses=responses[(self.sample_size // 2):])])
    
    bt.logging.info(f"Ranked responses: {responseRanks}")

    self.update_scores(responseRanks, miner_uids)
