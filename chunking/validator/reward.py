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

from typing import List
from chunking.protocol import chunkSynapse
from random import sample
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import bittensor as bt


def reward(self, document: str, chunk_size: int, response: chunkSynapse) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    if not response.chunks:
        print("no response")
        return 0
    chunks = response.chunks
    reward = 0.0
    smallChunks = []
    size_penalty = 0
    document_words = word_tokenize(document)
    combined_chunk_words = ''
    for i in range(len(chunks)):

        # check that every word in chunk exists and is in the same order as the source document
        chunk_words = ' '.join(word_tokenize(chunks[i]))
        combined_chunk_words += ' ' + chunk_words
        if chunk_words not in ' '.join(document_words):
            return 0

        # add up size penalty to be applied later
        chunk_length = len(chunks[i])
        if chunk_length > chunk_size:
            size_penalty += ((chunk_length / chunk_size) - 1) * 10

        # create test segments
        sentences = sent_tokenize(chunks[i])
        for j in range(0, len(sentences), 3):
            text = " ".join(sentences[j:j+3])
            smallChunks.append(smallChunk(i, text))

    # check that every set of 3 adjacent words from the document appears in the chunks
    for i in range(0, len(document_words), 3):
        if (len(' '.join(document_words[i:i+3])) < chunk_size
            and ' '.join(document_words[i:i+3]) not in combined_chunk_words):
            return 0

    # pick out segments to use for evaluation
    if self.num_embeddings < len(smallChunks):
        testChunks = sample(smallChunks, self.num_embeddings)
    else:
        testChunks = smallChunks

    # calculate rewards using embeddings of test chunks
    embeddings = self.client.embeddings.create(
        input=[testChunk.text for testChunk in testChunks],
        model="text-embedding-ada-002"
    ).data
    embeddings = [item.embedding for item in embeddings]
    for i in range(len(testChunks) - 1):
        j = i + 1
        while j < len(testChunks):
            if testChunks[i].sourceChunk == testChunks[j].sourceChunk:
                reward += np.dot(np.asarray(embeddings[i]), np.asarray(embeddings[j]))
            else:
                reward -= np.dot(np.asarray(embeddings[i]), np.asarray(embeddings[j]))
            j += 1

    # calculate and return final reward
    reward = 1.01 ** reward # ensures that all rewards are positive
    if response.dendrite.process_time > response.time_soft_max:
        reward *= (2/3) ** (response.dendrite.process_time - response.time_soft_max)
    reward *= (2/3) ** size_penalty
    return reward

def get_rewards(
        self,
        document: str,
        chunk_size: int,
        responses: List[chunkSynapse],
) -> np.ndarray:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.

    Returns:
    - np.ndarray: 
    """
    # Get all the reward results by iteratively calling your reward() function.
    rewards = np.array([float(reward(self, document, chunk_size, response)) for response in responses])
    return rewards

def rank_responses(
        rewards: np.ndarray,
) -> np.ndarray:
    """
    Returns an array containing the ranks of the responses using their rewards.

    Args:
    - rewards (List[float]): The list of rewards that were calculated.

    Returns:
    - np.ndarray: 
    """

    response_ranks = np.zeros_like(rewards)

    for i in range(len(rewards)):
        response_ranks[rewards.argmax()] = i
        rewards[rewards.argmax()] = -np.inf
    return response_ranks
    
class smallChunk():
    def __init__(self, sourceChunk, text):
        self.sourceChunk = sourceChunk
        self.text = text
