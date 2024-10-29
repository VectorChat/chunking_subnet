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

import hashlib
from math import ceil, e
import time
from typing import List, Tuple

from openai import AsyncOpenAI, OpenAI
from chunking.protocol import chunkSynapse
from random import sample
from nltk.tokenize import sent_tokenize, wordpunct_tokenize
import numpy as np
from aiomultiprocess import Pool

from chunking.utils.tokens import num_tokens_from_string
import bittensor as bt
import regex as re
import dill


PUNCTUATION_REGEX = r'([.,!?"\'])'


def custom_word_tokenize(text: str) -> List[str]:
    initial_words = wordpunct_tokenize(text)

    words_str = " ".join(initial_words)

    # add spaces around punctuation
    words_str = re.sub(PUNCTUATION_REGEX, r" \1 ", words_str)

    # remove extra spaces
    words_str = re.sub(r"\s+", " ", words_str).strip()

    return words_str.split()


def check_chunk_words_in_document(chunk: str, document: str, verbose: bool = False):
    def _verbose(msg: str):
        if verbose:
            print(msg)

    start_time = time.time()
    chunk_words = custom_word_tokenize(chunk)

    document_words = custom_word_tokenize(document)

    chunk_words_str = " ".join(chunk_words).strip()
    document_words_str = " ".join(document_words).strip()

    if chunk_words_str in document_words_str:
        _verbose(
            f"Time to check chunk words in document words: {time.time() - start_time} seconds"
        )
        return True
    else:
        _verbose(
            f"Time to check chunk words not in document words: {time.time() - start_time} seconds"
        )

        if verbose:
            closest_match_index = 0
            highest_matches_so_far = 0
            for i in range(len(document_words) - len(chunk_words) + 1):
                num_matches = 0
                for j in range(len(chunk_words)):
                    if document_words[i + j] == chunk_words[j]:
                        num_matches += 1
                    else:
                        break
                if num_matches > highest_matches_so_far:
                    highest_matches_so_far = num_matches
                    closest_match_index = i

            BLUE = "\033[94m"
            YELLOW = "\033[93m"
            RED = "\033[91m"
            ENDC = "\033[0m"

            closest_match_end_index = closest_match_index + len(chunk_words)

            closest_match_words = document_words[
                closest_match_index:closest_match_end_index
            ]

            chunk_str = ""
            closest_match_str_document = ""
            for i in range(len(chunk_words)):
                if chunk_words[i] == closest_match_words[i]:
                    chunk_str += " " + BLUE + chunk_words[i] + ENDC
                    closest_match_str_document += (
                        " " + BLUE + closest_match_words[i] + ENDC
                    )
                else:
                    chunk_str += " " + RED + chunk_words[i] + ENDC
                    closest_match_str_document += (
                        " " + YELLOW + closest_match_words[i] + ENDC
                    )

            print("=" * 100)
            print(
                f"Unable to find exact match for chunk words:\n\nClosest match:\n{chunk_str}\n\nDocument:\n{closest_match_str_document}"
            )
            print("=" * 100)
        return False


def check_document_words_in_chunks(
    document: str, chunks: List[str], chunk_size: int, k=3
):
    document_words = custom_word_tokenize(document)
    combined_chunk_words = ""
    for chunk in chunks:
        combined_chunk_words += " " + " ".join(custom_word_tokenize(chunk))

    combined_chunk_words = combined_chunk_words.strip()

    for i in range(0, len(document_words), k):
        document_words_str = " ".join(document_words[i : i + k])
        if (
            len(document_words_str) < chunk_size
            and document_words_str not in combined_chunk_words
        ):
            print(
                f"Unable to find {document_words_str} in combined_chunk_words, k: {k}"
            )
            return False

    return True


async def reward(
    document: str,
    chunk_size: int,
    chunk_qty: int,
    response: chunkSynapse,
    num_embeddings: int,
    client: AsyncOpenAI | None = None,
    verbose: bool = False,
) -> Tuple[float, dict]:
    """
    Reward the miner based on the chunks they make for a specific document.

    The reward function checks that:
    - every word in each chunk exists in the source document
    - every set of 3 adjacent words in the document appears in at least one chunk

    If these conditions are not met, the reward is set to 0.

    Exponential penalties are applied for:
    - excessive chunk size
    - excessive number of chunks
    - time above the time soft max

    It creates "smallChunks" from the chunks to be evaluated for quality. These are segments of the chunks that are 3 adjacent sentences long (currently).
    Then, "testChunks" are sampled (or the entire smallChunks if num_embeddings is less than the number of smallChunks) to be used for evaluation.

    The reward is calculated by taking the mean intrachunk similarity and subtracting the mean interchunk similarity.
    - _Intrachunk similarity_ is the dot product of the embeddings of the testChunks if they appeared in the _same chunk_.
    - _Interchunk similarity_ is the dot product of the embeddings of the testChunks if they appeared in _different chunks_.

    Args:
    - self (Validator): The validator object, used to get the OpenAI client and number of embeddings.
    - document (str): The document to be chunked.
    - chunk_size (int): The soft max size of a chunk in characters before penalties are applied.
    - chunk_qty (int): The soft max number of chunks before penalties are applied.
    - response (chunkSynapse): The synapse received from the miner.
    - client (AsyncOpenAI | None): An optional OpenAI client to use for embedding (useful for testing when a validator instance is not available)
    - num_embeddings (int | None): An optional number of embeddings to use for evaluation (useful for testing when a validator instance is not available)
    - verbose (bool): Whether to print verbose output.

    Returns:
    - Tuple[float, dict]: A tuple containing the reward and extra info (penalties, timing, etc.) for wandb logging.
    """

    # helper function to print verbose output
    def _verbose(msg: str):
        if verbose:
            print(msg)

    if client is None:
        client = AsyncOpenAI()

    # dictionary to store extra info (penalties, timing, etc.) for wandb logging
    extra_info_dict = {}

    # helper function to return early if there is an error
    def _get_early_return_stuff(msg: str):
        _verbose(msg)

        return 0, extra_info_dict

    if not response.chunks:
        return _get_early_return_stuff(
            f"No chunks found in response {response.name}, axon {response.axon.hotkey[:10] if response.axon is not None and response.axon.hotkey is not None else 'None'}"
        )

    chunks = response.chunks
    intrachunk_similarities = []
    interchunk_similarities = []
    smallChunks = []
    size_penalty = 0

    qty_penalty = 0

    # penalize an excessive number of chunks
    num_chunks = len(chunks)
    if num_chunks > chunk_qty:
        qty_penalty += 10 * ((num_chunks / chunk_qty) - 1) * 10
        _verbose(
            f"Too many chunks: {num_chunks} chunks, new quantity penalty: {qty_penalty}"
        )

    start_time = time.time()

    # check that every set of 3 adjacent words from the document appears in the chunks
    if not check_document_words_in_chunks(document, chunks, chunk_size):
        return _get_early_return_stuff(
            f"Every set of 3 adjacent words from the document does not appear in the chunks"
        )

    for i in range(len(chunks)):

        # check that every word in chunk exists and is in the same order as the source document
        if not check_chunk_words_in_document(chunks[i], document, verbose):
            return _get_early_return_stuff(
                f"Chunk {i} does not contain all words from the document"
            )

        # add up size penalty to be applied later
        chunk_length = len(chunks[i])
        if chunk_length > chunk_size:
            size_penalty += ((chunk_length / chunk_size) - 1) * 10
            _verbose(
                f"Chunk {i} is too long: {chunk_length} characters, new size penalty: {size_penalty}"
            )

        # create test segments
        sentences = sent_tokenize(chunks[i])
        for j in range(0, len(sentences), 3):
            text = " ".join(sentences[j : j + 3])
            smallChunks.append(smallChunk(i, text))

        _verbose(
            f"Chunk {i} has {len(sentences)} sentences. Added {ceil(len(sentences) / 3)} test segments"
        )

    end_time = time.time()
    print(f"Time to run checks: {end_time - start_time} seconds")

    _verbose(
        f"Passed: Every set of 3 adjacent words from the document appears in the chunks"
    )

    testChunks: list[smallChunk]

    # pick out segments to use for evaluation
    if num_embeddings < len(smallChunks):
        testChunks = sample(smallChunks, num_embeddings)
    else:
        testChunks = smallChunks

    _verbose(f"Using {len(testChunks)} test segments for evaluation")

    # all text to be embedded
    all_text = " ".join([testChunk.text for testChunk in testChunks])

    # calculate the number of tokens in the text (for logging/accounting purposes)
    num_tokens = num_tokens_from_string(all_text, "gpt-4o-mini")

    print(f"Using {num_tokens} tokens for test embeddings")
    start_time = time.time()

    # calculate rewards using embeddings of test chunks
    res = await client.embeddings.create(
        input=[testChunk.text for testChunk in testChunks],
        model="text-embedding-ada-002",
    )
    data = res.data
    embeddings = [item.embedding for item in data]

    end_time = time.time()
    print(f"Time to get embeddings: {end_time - start_time} seconds")

    start_time = time.time()

    # calculate intrachunk and interchunk similarities
    for i in range(len(testChunks) - 1):
        j = i + 1
        while j < len(testChunks):
            if testChunks[i].sourceChunk == testChunks[j].sourceChunk:
                intrachunk_similarities.append(
                    np.dot(np.asarray(embeddings[i]), np.asarray(embeddings[j]))
                )
            else:
                interchunk_similarities.append(
                    np.dot(np.asarray(embeddings[i]), np.asarray(embeddings[j]))
                )
            j += 1

    # calculate the embedding reward
    reward = (
        np.mean(intrachunk_similarities) if len(intrachunk_similarities) > 0 else 0
    ) - (np.mean(interchunk_similarities) if len(interchunk_similarities) > 0 else 0)

    end_time = time.time()
    print(f"Time to calculate embedding reward: {end_time - start_time} seconds")

    _verbose(f"Embedding reward: {reward}")
    _verbose(f"Size penalty: {size_penalty}")
    _verbose(f"Quantity penalty: {qty_penalty}")

    # store extra info for wandb logging/printing
    extra_info_dict["embeddings"] = embeddings
    extra_info_dict["intrachunk_similarities"] = intrachunk_similarities
    extra_info_dict["interchunk_similarities"] = interchunk_similarities
    extra_info_dict["size_penalty"] = size_penalty
    extra_info_dict["embedding_reward"] = reward
    extra_info_dict["qty_penalty"] = qty_penalty
    extra_info_dict["num_embed_tokens"] = num_tokens

    # ensure that the reward is positive
    reward = e**reward
    _verbose(f"Ensuring reward is positive (e ** reward):\n{reward}")

    # apply time penalty if the process time is greater than the time soft max
    if (
        response.dendrite
        and response.dendrite.process_time
        and response.dendrite.process_time > response.time_soft_max
    ):
        over_time = response.dendrite.process_time - response.time_soft_max
        _verbose(f"Applying time penalty: {over_time} seconds over time")
        time_penalty = (2 / 3) ** over_time

        extra_info_dict["time_penalty"] = time_penalty

        # apply time penalty to the reward
        reward *= time_penalty

    # apply size and quantity penalties
    reward *= (2 / 3) ** (size_penalty + qty_penalty)

    return reward, extra_info_dict


def get_chunk_hash(chunk: str) -> str:
    return hashlib.sha256(chunk.encode()).hexdigest()


def get_chunks_hash(chunks: List[str] | None) -> str:
    if chunks is None:
        return ""

    if len(chunks) == 0:
        return ""

    final_hash = get_chunk_hash(re.sub(r"\s+", " ", chunks[0]).strip())
    for chunk in chunks[1:]:
        final_hash += get_chunk_hash(re.sub(r"\s+", " ", chunk).strip())

    return final_hash


async def get_rewards(
    document: str,
    chunk_size: int,
    chunk_qty: int,
    responses: List[chunkSynapse],
    client: AsyncOpenAI,
    num_embeddings: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Get the rewards for the given query and responses, returning the rewards and extra info (penalties, timing, etc.) for each response.

    Args:
    - document (str): The document to be chunked.
    - chunk_size (int): The soft max size of a chunk in characters before penalties are applied.
    - chunk_qty (int): The soft max number of chunks before penalties are applied.
    - responses (List[chunkSynapse]): A list of responses from the miner.

    Returns:
    - np.ndarray: An array of rewards for each response.
    - List[dict]: A list of extra info (penalties, timing, etc.) for each response.
    """

    rewards = np.zeros(len(responses))
    extra_infos = []

    chunks_hash_to_info = {}
    hashes = []

    async with Pool() as pool:
        for response in responses:
            miner_hotkey = response.axon.hotkey or "not found"
            print(f"handling response from {miner_hotkey[:10]}")
            chunks_hash = (
                get_chunks_hash(response.chunks) if response is not None else ""
            )
            if chunks_hash not in hashes and response is not None:
                try:
                    print(
                        f"calculating reward for new chunks hash: {chunks_hash[:10]}..., there are {len(response.chunks)} chunks"
                    )
                    # reward_value, extra_info = await reward(
                    #     document,
                    #     chunk_size,
                    #     chunk_qty,
                    #     response,
                    #     client=client,
                    #     num_embeddings=num_embeddings,
                    #     verbose=verbose,
                    # )
                    print("calling reward() via aiomultiprocess pool")
                    reward_value, extra_info = await pool.apply(
                        reward,
                        kwds={
                            "document": document,
                            "chunk_size": chunk_size,
                            "chunk_qty": chunk_qty,
                            "response": response,
                            "num_embeddings": num_embeddings,
                            "verbose": verbose,
                            # no client to avoid pickling
                        },
                    )
                except Exception as e:
                    print(
                        f"Error calculating reward for response {response.name}, axon {miner_hotkey[:10]}: {e}"
                    )
                    reward_value = 0
                    extra_info = {}

                chunks_hash_to_info[chunks_hash] = {
                    "reward": reward_value,
                    "extra_info": extra_info,
                }
                print(
                    f"calculated reward for new chunks hash: {chunks_hash[:10]}..., reward: {reward_value}"
                )
            hashes.append(chunks_hash)

    for i, response in enumerate(responses):
        chunks_hash = hashes[i]

        chunks_info = chunks_hash_to_info.get(chunks_hash)

        if chunks_info:
            rewards[i] = float(chunks_info["reward"])
            extra_infos.append(chunks_info["extra_info"])
        else:
            rewards[i] = 0
            extra_infos.append({})
        print(
            f"response {i}: {rewards[i]} - {response.axon.hotkey[:10] if response.axon is not None and response.axon.hotkey is not None else 'None'} - {chunks_hash[:10]}..."
        )

    # print(f"rewards is picklable: {dill.pickles(rewards)}")
    # print(f"extra_infos is picklable: {dill.pickles(extra_infos)}")

    return rewards, extra_infos


def rank_responses(
    rewards: np.ndarray[np.float64],
) -> np.ndarray[np.int32]:
    """
    Returns an array containing the ranks of the responses using their rewards. Higher reward is better.    Higher reward is better. Ties are broken by shorter process time.

    Args:
    - rewards (np.ndarray): The array of rewards that were calculated.

    Returns:
    - np.ndarray: Array of ranks for each response.
    """

    reward_to_count = {}
    for reward in rewards:
        if reward in reward_to_count:
            reward_to_count[reward] += 1
        else:
            reward_to_count[reward] = 1

    reward_to_rank = {}
    rank = 0
    for reward in sorted(reward_to_count.keys(), reverse=True):
        reward_to_rank[reward] = rank
        rank += reward_to_count[reward]

    response_ranks = np.zeros_like(rewards)
    for i, reward in enumerate(rewards):
        if reward == 0:
            response_ranks[i] = -1
        else:
            response_ranks[i] = reward_to_rank[reward]

    return np.array(response_ranks)


def rank_responses_global(
    self,
    group_rank_values: np.ndarray[np.float64],
    ranked_responses: np.ndarray[int],
    miner_group_uids: np.ndarray[int],
    override_scores: np.ndarray[float] | None = None,
) -> np.ndarray[np.float64]:
    # inf means the response should not be ranked
    ranked_responses_global = np.full_like(ranked_responses, np.inf)

    scores = override_scores if override_scores is not None else self.scores

    ranked_responses = ranked_responses.astype(int)

    # loop through the ranked responses and assign a global rank to each response
    for i, rank in enumerate(ranked_responses):
        if rank != -1:
            bt.logging.debug(f"rank: {rank}, group_rank_values: {group_rank_values}")
            rank_value = group_rank_values[rank]
            ranked_responses_global[i] = rank_value
        elif not np.isinf(scores[miner_group_uids[i]]):
            # give response worst rank in the group
            ranked_responses_global[i] = group_rank_values[-1]

    ranked_responses_global = ranked_responses_global.astype(np.float64)

    return ranked_responses_global


class smallChunk:
    def __init__(self, sourceChunk: str, text: str):
        self.sourceChunk = sourceChunk
        self.text = text
