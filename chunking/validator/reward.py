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

import difflib
from math import ceil, e
from typing import List, Tuple

from openai import OpenAI
from chunking.protocol import chunkSynapse
from random import sample
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
import numpy as np

from chunking.validator.task_api import num_tokens_from_string
from neurons.validator import Validator
import bittensor as bt
import regex as re

PUNCTUATION_REGEX = r'([.,!?"\'])'


def custom_word_tokenize(text: str) -> List[str]:
    initial_words = wordpunct_tokenize(text)

    final_words = []

    for word in initial_words:
        final_word = ""
        for char in word:
            if re.match(PUNCTUATION_REGEX, char):
                final_word += " " + char + " "
            else:
                final_word += char

        # remove extra spaces
        final_word = re.sub(r"\s+", " ", final_word).strip()

        final_words.append(final_word)

    return final_words


def check_chunk_words_in_document(chunk: str, document: str, verbose: bool = False):
    def _verbose(msg: str):
        if verbose:
            print(msg)

    # word_tokenizer = TreebankWordTokenizer()
    chunk_words = wordpunct_tokenize(chunk)
    document_words = wordpunct_tokenize(document)

    _verbose(f"created {len(chunk_words)} chunk words")
    _verbose(f"created {len(document_words)} document words")

    chunk_words_str = " ".join(chunk_words)
    document_words_str = " ".join(document_words)

    # Regex to match any punctuation
    punctuation_regex = r'([.,!?"\'])'

    # Add space before and after each punctuation mark
    chunk_words_str = re.sub(punctuation_regex, r" \1 ", chunk_words_str)
    document_words_str = re.sub(punctuation_regex, r" \1 ", document_words_str)

    _verbose("removed punctuation")

    # Remove extra spaces
    chunk_words_str = re.sub(r"\s+", " ", chunk_words_str).strip()
    document_words_str = re.sub(r"\s+", " ", document_words_str).strip()

    _verbose("removed extra spaces")

    if chunk_words_str in document_words_str:
        _verbose("chunk words in document words")
        return True
    else:
        _verbose("chunk words not in document words")

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
            # print("-" * 100)
            # print(f"{YELLOW} chunk words: {chunk_words} {ENDC}")
            # print(f"{BLUE} document words: {document_words} {ENDC}")
            # print("-" * 100)
            # print(f"{YELLOW} chunk: {chunk} {ENDC}")
            # print(f"{BLUE} document: {document} {ENDC}")
            print("=" * 100)
        return False


def check_document_words_in_chunks(
    document: str, chunks: List[str], chunk_size: int, k=3
):
    document_words = custom_word_tokenize(document)
    combined_chunk_words = " "
    for chunk in chunks:
        combined_chunk_words += " " + " ".join(custom_word_tokenize(chunk))

    for i in range(0, len(document_words), k):
        document_words_str = " ".join(document_words[i : i + k])
        if (
            len(document_words_str) < chunk_size
            and document_words_str not in combined_chunk_words
        ):
            return False

    return True


def reward(
    self: Validator | None,
    document: str,
    chunk_size: int,
    chunk_qty: int,
    response: chunkSynapse,
    override_client: OpenAI | None = None,
    override_num_embeddings: int | None = None,
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
    - override_client (OpenAI | None): An optional OpenAI client to use for embedding (useful for testing when a validator instance is not available)
    - override_num_embeddings (int | None): An optional number of embeddings to use for evaluation (useful for testing when a validator instance is not available)
    - verbose (bool): Whether to print verbose output.

    Returns:
    - Tuple[float, dict]: A tuple containing the reward and extra info (penalties, timing, etc.) for wandb logging.
    """

    if not self and not override_client and not override_num_embeddings:
        raise Exception(
            "Either self or override_client and override_num_embeddings must be provided"
        )

    # helper function to print verbose output
    def _verbose(msg: str):
        if verbose:
            print(msg)

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

    for i in range(len(chunks)):

        # check that every word in chunk exists and is in the same order as the source document
        if not check_chunk_words_in_document(chunks[i], document):
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

    # check that every set of 3 adjacent words from the document appears in the chunks
    if not check_document_words_in_chunks(document, chunks, chunk_size):
        return _get_early_return_stuff(
            f"Every set of 3 adjacent words from the document does not appear in the chunks"
        )

    _verbose(f"Passed: Every set of 3 adjacent words from the document appears in the chunks")

    num_embeddings = (
        override_num_embeddings if override_num_embeddings else self.num_embeddings
    )

    testChunks: list[smallChunk]

    # pick out segments to use for evaluation
    if num_embeddings < len(smallChunks):
        testChunks = sample(smallChunks, num_embeddings)
    else:
        testChunks = smallChunks

    _verbose(f"Using {len(testChunks)} test segments for evaluation")

    client = override_client if override_client else self.client

    # all text to be embedded
    all_text = " ".join([testChunk.text for testChunk in testChunks])

    # calculate the number of tokens in the text (for logging/accounting purposes)
    num_tokens = num_tokens_from_string(all_text, "o200k_base")

    bt.logging.info(f"Using {num_tokens} tokens for test embeddings")

    # calculate rewards using embeddings of test chunks
    embeddings = client.embeddings.create(
        input=[testChunk.text for testChunk in testChunks],
        model="text-embedding-ada-002",
    ).data
    embeddings = [item.embedding for item in embeddings]

    # calculate intrachunk and interchunk similarities
    _verbose(f"Calculated embeddings for {len(embeddings)} test segments")
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


def get_rewards(
    self,
    document: str,
    chunk_size: int,
    chunk_qty: int,
    responses: List[chunkSynapse],
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

    for i, response in enumerate(responses):
        try:
            # skip responses that have no chunks
            if not response.chunks or len(response.chunks) == 0:
                raise Exception(
                    f"No chunks found in response {response.name}, axon {response.axon.hotkey[:10]}"
                )

            # get the reward for the response
            reward_value, extra_info = reward(
                self, document, chunk_size, chunk_qty, response
            )

            # store the reward for the response
            rewards[i] = float(reward_value)
            extra_infos.append(extra_info)
        except Exception as e:
            # if there is an error, log it and set the reward to 0
            print(
                f"Error calculating reward for response {response.name}, axon {response.axon.hotkey[:10]}: {e}"
            )
            rewards[i] = 0
            extra_infos.append({})

    return rewards, extra_infos


def rank_responses(
    rewards: np.ndarray,
) -> np.ndarray:
    """
    Returns an array containing the ranks of the responses using their rewards. Higher reward is better.    Higher reward is better. Ties are broken by shorter process time.

    Args:
    - rewards (np.ndarray): The array of rewards that were calculated.

    Returns:
    - np.ndarray: Array of ranks for each response.
    """

    response_ranks = np.zeros_like(rewards)

    rank = 0
    for _ in range(len(rewards)):
        next_best_index = rewards.argmax()

        if rewards[next_best_index] == 0:
            # should not be ranked
            response_ranks[next_best_index] = -1
        else:
            response_ranks[next_best_index] = rank
            rank += 1

        rewards[next_best_index] = -np.inf
    return response_ranks


class smallChunk:
    def __init__(self, sourceChunk: str, text: str):
        self.sourceChunk = sourceChunk
        self.text = text
