import asyncio
import random
import time
from typing import Literal, Optional, List, Tuple
from venv import logger
import aiohttp
import bittensor as bt
from openai import AsyncOpenAI, OpenAI
from chunking.protocol import chunkSynapse, chunkSynapseType
import requests
import numpy as np
from sr25519 import sign
import json
import os
from random import choice, choices
from math import ceil

from chunking.utils.chunks import calculate_chunk_qty
from chunking.utils.synthetic.synthetic import generate_synthetic_synapse
from chunking.utils.tokens import num_tokens_from_string
from chunking.validator.types import TaskType
from chunking.utils.relay.relay import make_relay_payload


class Task:
    """
    A class to represent a task that the validator can assign to a miner.
    """

    def __init__(
        self,
        synapse: chunkSynapse,
        task_type: TaskType,
        task_id: int,
        page_id: int = -1,
        miner_uids: Optional[List[int]] = None,  # used for external organic task api
    ):
        self.synapse = synapse
        self.task_type = task_type
        self.task_id = task_id
        self.miner_uids = miner_uids
        self.page_id = page_id

    @classmethod
    def get_organic_task(cls, validator) -> Optional["Task"]:
        """
        Get an organic task from the API host.

        Args:
            validator (Validator): The validator instance requesting the task.

        Returns:
            Task: The organic task
        """
        hotkey = validator.wallet.get_hotkey()
        nonce = time.time_ns()
        data = {"hotkey_address": hotkey.ss58_address, "nonce": nonce}

        bt.logging.debug(
            f"Requesting task from API host: '{os.environ['CHUNKING_API_HOST']}'"
        )
        bt.logging.debug(f"Request body: {data}")

        # sign request with validator hotkey
        request_signature = sign(
            (hotkey.public_key, hotkey.private_key), str.encode(json.dumps(data))
        ).hex()

        API_host = os.environ["CHUNKING_API_HOST"]
        task_url = f"{API_host}/task_api/get_new_task/"
        headers = {"Content-Type": "application/json"}
        request_data = {"data": data, "signature": request_signature}
        bt.logging.debug(f"Request data: {request_data}")

        try:
            response = requests.post(url=task_url, headers=headers, json=request_data)
            if response.status_code == 502:
                raise Exception(f"API Host: '{API_host}' is down")
            elif response.status_code == 403:
                raise Exception(response.text())
            elif response.status_code != 200:
                raise Exception(
                    f"Post to API failed with status code: {response.status_code}"
                )
            else:
                task = response.json()
                if task["task_id"] != -1:
                    task_id = task["task_id"]
                    miner_uids = task.get("miner_uids")
                    bt.logging.debug(f"Received organic query with task id: {task_id}")
                    if task.get("time_soft_max") == None:
                        task["time_soft_max"] = 5.0
                    if task.get("chunk_size") == None:
                        task["chunk_size"] = 4096
                    if task.get("chunk_qty") == None:
                        task["chunk_qty"] = ceil(
                            ceil(len(task["document"]) / task["chunk_size"]) * 1.5
                        )
                    bt.logging.debug(f"task: {task}")

                    synapse = chunkSynapse(
                        document=task["document"],
                        time_soft_max=float(task["time_soft_max"]),
                        chunk_size=int(task["chunk_size"]),
                        chunk_qty=int(task["chunk_qty"]),
                    )
                else:
                    bt.logging.info(
                        f"No organic task available. Generating synthetic query"
                    )
                    raise Exception("No organic task available")
                return (
                    Task(
                        synapse=synapse,
                        task_type="organic",
                        task_id=task_id,
                        miner_uids=miner_uids,
                    ),
                )

        except Exception as e:
            bt.logging.error(
                f"Failed to get task from API host: '{API_host}'. Exited with exception\n{e}"
            )
            return None

    @classmethod
    async def get_synthetic_task(cls, validator) -> "Task":
        """
        Get a synthetic task from the API host.

        Args:
            validator (Validator): The validator instance requesting the task.

        Returns:
            Task: The synthetic task
        """
        bt.logging.debug("Getting synthetic task")
        synapse, pageid = await generate_synthetic_synapse(validator)
        return Task(synapse=synapse, task_type="synthetic", task_id=-1, page_id=pageid)

    @classmethod
    async def get_new_task(self, validator) -> "Task":
        """
        Get a new task based on the validator's config.

        This function returns either an organic or synthetic task.

        If the validator is configured to allow organic tasks, it will request an organic task from the API host.
        If no organic task is available, it will generate a synthetic task.

        If the validator is not configured to allow organic tasks, it will generate a synthetic task regardless.

        Args:
            validator (Validator): The validator instance requesting the task.

        Returns:
            Tuple[Task, int]: A tuple containing the task and the page ID.
        """
        if os.environ.get("ALLOW_ORGANIC_CHUNKING_QUERIES") == "True":
            task = self.get_organic_task(validator)
        else:
            task = None

        if task is None:
            task = await self.get_synthetic_task(validator)

        bt.logging.debug(f"Created task: {task}")

        try:
            # Make relay payload and pin to IPFS for both organic and synthetic queries
            CID = await make_relay_payload(
                task.synapse.document, validator.aclient, validator.wallet
            )
        except Exception as e:
            bt.logging.error(f"Failed to pin document to IPFS: {e}")
            CID = None

        task.synapse.CID = CID

        bt.logging.debug(f"Added CID: {CID} to task")

        return task

    @classmethod
    def return_response(cls, validator, response_data):
        """
        Return a miner response with chunks to the API host.

        Args:
            validator (Validator): The validator instance returning the response.
            response_data (dict): The data to be returned to the API host.
        """
        validator_hotkey = validator.wallet.get_hotkey()
        validator_sig = sign(
            (validator_hotkey.public_key, validator_hotkey.private_key),
            str.encode(json.dumps(response_data)),
        ).hex()
        API_host = os.environ["CHUNKING_API_HOST"]
        task_url = f"{API_host}/task_api/organic_response/"
        headers = {"Content-Type": "application/json"}
        data = {
            "response_data": response_data,
            "validator_signature": validator_sig,
        }
        try:
            response = requests.post(task_url, headers=headers, json=data)
        except Exception as e:
            bt.logging.error(
                f"Failed to return response to API host: '{API_host}'. Exited with exception\n{e}"
            )

    @classmethod
    def upload_logs(cls, validator, log_data):
        """
        Upload logs to the API host.

        Args:
            validator (Validator): The validator instance uploading the logs.
            log_data (dict): The data to be uploaded to the API host.
        """
        hotkey = validator.wallet.get_hotkey()
        signature = sign(
            (hotkey.public_key, hotkey.private_key), str.encode(json.dumps(log_data))
        ).hex()

        API_host = os.environ["CHUNKING_API_HOST"]
        task_url = f"{API_host}/task_api/log/"
        headers = {"Content-Type": "application/json"}
        data = {
            "log_data": log_data,
            "signature": signature,
        }
        try:
            response = requests.post(task_url, headers=headers, json=data)
            bt.logging.debug(f"upload_logs: response: {response.status_code}")

            if response.status_code == 502:
                raise Exception(f"API Host: '{API_host}' is down")
            elif response.status_code == 403:
                raise Exception(response.text())
            elif response.status_code != 200:
                raise Exception(
                    f"Post to API failed with status code: {response.status_code}"
                )
            else:
                bt.logging.debug(
                    f"Successfully uploaded logs to API host: '{API_host}'"
                )

        except Exception as e:
            bt.logging.error(
                f"Failed to upload logs to API host: '{API_host}'. Exited with exception\n{e}"
            )
