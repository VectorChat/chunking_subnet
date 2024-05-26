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

import torch
from typing import List
from chunking.protocol import chunkSynapse
from multiprocessing import Pool
import random
import numpy as np
from langchain.text_splitter import NLTKTextSplitter
import bittensor as bt

def reward(document: str, response: chunkSynapse) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    if not response:
        bt.logging.info("No response")
        return 0
    reward = 0
    smallChunks = []
    splitter =  NLTKTextSplitter()
    sourceSentences = splitter.split_text(document)[0].split('\n\n')
    responseSentences = splitter.split_text(" ".join(response.chunks))[0].split('\n\n')

    for sentence in sourceSentences:
        if not sentence in responseSentences:
            bt.logging.info("mismatch")
            return 0
    for sentence in responseSentences:
        if not sentence in sourceSentences:
            bt.logging.info("mismatch")
            return 0
        
    for i in range(len(response.chunks)):
        sentences = splitter.split_text(response.chunks[i])[0].split('\n\n')
        for j in range(-(len(sentences) // -3)):
            if (j * 3 + 2) < len(sentences):
                text = " ".join([sentences[j * 3], sentences[j * 3 + 1], sentences[j * 3 + 2]])
            else:
                text = " ".join(sentences[j*3:])
            smallChunks.append(smallChunk(i, text))
    testChunks = sample(smallChunks, self.numEmbeddings)
    embeddings = self.client.embeddings.create(input=[testChunk.text for testChunk in testChunks], model="text-embedding-ada-002")
    for i in range(len(testChunks) - 1):
        j = i + 1
        while j < len(testChunks):
            if testChunks[i].sourceChunk == testChunks[j].sourceChunk:
                reward += np.dot(np.asarray(embeddings[i]), np.asarray(embeddings[j]))
            else:
                reward -= np.dot(np.asarray(embeddings[i]), np.asarray(embeddings[j]))
            j += 1
    if response.time_elapsed > 2.75:
        reward *= (2/3) ** (response.time_elapsed - 2.75)
    return reward


def get_rewards(
        self,
        document: str,
        responses: List[chunkSynapse],
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """
    # Get all the reward results by iteratively calling your reward() function.
    return torch.FloatTensor(
        [reward(document, response) for response in responses]
    ).to(self.device)



class smallChunk():
    def __init__(self, sourceChunk, text):
        self.sourceChunk = sourceChunk
        self.text = text
