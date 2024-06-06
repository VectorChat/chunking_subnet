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
import torch
from chunking.protocol import chunkSynapse
from chunking.validator.reward import get_rewards
from chunking.utils.uids import get_random_uids
import requests

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

    try:
        # don't grab miner ids if they are passed in from the organic axon request
        if not synapse or len(synapse.miner_uids) == 0:
            miner_uids = get_random_uids(self, k=min(self.config.neuron.sample_size, self.metagraph.n.item()))
            bt.logging.info(f"Getting miner uids: {miner_uids}")
    except Exception as e:
        #bt.logging.warning(f"Trouble setting miner_uids: {e}")
        bt.logging.warning("Defaulting to 1 uid, k=1 ... likely due to low # of miners available.")
        miner_uids = get_random_uids(self, k=1)

    if synapse:
        bt.logging.debug("Synapse was passed in")
        if len(synapse.miner_uids) > 0:
            miner_uids = synapse.miner_uids
            bt.logging.info(f"Parsed uids: {miner_uids}")
        
        if not synapse.timeout:
            synapse.timeout = 10.0
        
        if not synapse.maxTokensPerChunk:
            synapse.maxTokensPerChunk = 200

    else:
        page = choice([312990, 9046237, 585013, 444081, 12559806, 30873232, 9236, 9577500, 21501970])
        # page = requests.get('https://en.wikipedia.org/w/api.php', params={
        #     'action': 'query',
        #     'format': 'json',
        #     'list': 'random',
        #     'rnnamespace': 0,
        # }).json()['query']['random'][0]['id']

        document = requests.get('https://en.wikipedia.org/w/api.php', params={
            'action': 'query',
            'format': 'json',
            'pageids': page,
            'prop': 'extracts',
            'explaintext': True,
            'exsectionformat': 'plain',
            }).json()['query']['pages'][str(page)]['extract']
        document = document.replace("\n", " ").replace("\t", " ")
        document = ' '.join(document.split())
        synapse = chunkSynapse(document=page, timeout=30.0, maxTokensPerChunk=200)
        # The dendrite client queries the network.
    
    # bt.logging.debug(
    #             f"Querying axons: {[(uid, self.metagraph.axons[uid] )for uid in miner_uids]}"
    #         )

    responses = self.dendrite.query(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=False,
        timeout=synapse.timeout,
    )
    bt.logging.info("Received responses:") 
    for response in responses:
        if response.chunks:
            bt.logging.info(f"\t{[(chunk[:20] + '...') for chunk in response.chunks]}")
        else:
            bt.logging.info("No response")
    rewards = get_rewards(self, document=document, responses=responses)

    bt.logging.info(f"Scored responses: {rewards}")

    self.update_scores(rewards, miner_uids)
