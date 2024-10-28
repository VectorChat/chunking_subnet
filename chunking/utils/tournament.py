from tabulate import tabulate
import bittensor as bt
from chunking.protocol import chunkSynapse
import numpy as np
import json
import gzip
import base64

from chunking.utils.integrated_api.chunk.types import BenchmarkID, ChunkRequestType
from chunking.utils.log import debug_log_dict
from chunking.validator.task_api import Task


def pretty_print_rewards(
    miner_group_uids: list[int], rewards: list[float], extra_infos: list[dict]
):
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


def print_responses(responses: list[chunkSynapse]):
    # function to print a miner's response, useful for debugging
    bt.logging.debug("Responses:")
    for response in responses:
        print_response(response)


def compress_and_encode_string(string: str) -> str:
    compressed = gzip.compress(string.encode())
    encoded = base64.b64encode(compressed).decode()
    return encoded


def make_wandb_data(
    block_number: int,
    miner_group_uids: list[int],
    miner_group_index: int,
    task: Task,
    responses: list[chunkSynapse],
    rewards: list[float],
    reward_extra_infos: list[dict],
    ranked_responses: list[int],
    ranked_responses_global: list[float],
    alpha: float,
    request_type: ChunkRequestType,
    cur_scores: list[float],
    cur_rankings: list[int],
    benchmark_id: BenchmarkID | None,
    is_debug: bool = False,
) -> dict:
    # initial structure for wandb logging
    wandb_data = {
        "modality": "text",
        "sync_block": block_number,
        "task_type": task.task_type,
        "type": request_type.value,
        "benchmark_id": str(benchmark_id),
        "params": {
            "chunk_size": task.synapse.chunk_size,
            "chunk_qty": task.synapse.chunk_qty,
            "timeout": task.synapse.timeout,
            "time_soft_max": task.synapse.time_soft_max,
            "CID": task.synapse.CID,
            "document": compress_and_encode_string(task.synapse.document),
        },
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
            "uids": [],
            "hotkeys": [],
            "miner_group_index": miner_group_index,
            "alpha": alpha,
        },
        "page_id": task.page_id or -1,
    }

    # if benchmark, socres/rankings will not be added in update_scores func so add cur rankings/scores manualy
    if request_type == ChunkRequestType.benchmark:
        for uid in range(len(cur_scores)):
            wandb_data["all"]["scores"][str(uid)] = cur_scores[uid]

        for rank_i in range(len(cur_rankings)):
            uid = cur_rankings[rank_i]
            wandb_data["all"]["rankings"][str(uid)] = rank_i

    # log the uids that are part of the miner group that is queried
    wandb_data["group"]["uids"] = miner_group_uids

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
        encoded_chunks = compress_and_encode_string(json_str)

        wandb_data["group"]["chunks"][str(uid)] = encoded_chunks

    # list of hotkeys from the responses
    hotkeys = [response.axon.hotkey for response in responses]

    # log the hotkeys from the responses for wandb logging
    wandb_data["group"]["hotkeys"] = hotkeys

    miner_group_cur_scores = [cur_scores[uid] for uid in miner_group_uids]

    for i in range(len(miner_group_uids)):
        uid = str(miner_group_uids[i])
        wandb_data["group"]["rewards"][uid] = rewards[i]
        wandb_data["group"]["local_rankings"][uid] = ranked_responses[i]
        wandb_data["group"]["global_rankings"][uid] = ranked_responses_global[i]
        wandb_data["group"]["scores"][uid] = miner_group_cur_scores[i]

    if is_debug:
        bt.logging.debug("WANDB DATA:")
        debug_log_dict(wandb_data, truncate=100, indent=2)

    return wandb_data
