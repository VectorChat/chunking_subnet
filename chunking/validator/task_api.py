from typing import Optional, List
import bittensor as bt
from chunking.protocol import chunkSynapse
import requests
import numpy as np
from sr25519 import sign
import json
import os
from random import choice

class Task():
    def __init__(
        self,
        synapse: chunkSynapse,
        task_type: str,
        task_id: int,
        miner_uids: Optional[List[int]] = None,
    ):
        self.synapse = synapse
        self.task_type = task_type
        self.task_id = task_id
        self.miner_uids = miner_uids
    @classmethod
    def get_new_task(self, validator):

        if os.environ.get('ALLOW_ORGANIC_CHUNKING_QUERIES') == 'True':
            hotkey = validator.wallet.get_hotkey()
            nonce = validator.step
            data = {
                'hotkey_address': hotkey.ss58_address,
                'nonce': nonce
            }

            # sign request with validator hotkey
            request_signature = sign(
                (hotkey.public_key, hotkey.private_key),
                str.encode(json.dumps(data))
                ).hex()

            API_host = os.environ['CHUNKING_API_HOST']
            task_url = f"{API_host}/task_api/get_new_task/"
            headers = {"Content-Type": "application/json"}
            request_data = {
                'data': data, 
                'signature': request_signature
                }
            try:
                response = requests.post(url=task_url, headers=headers, json=request_data)
                if response.status_code == 502:
                    raise Exception(f"API Host: \'{API_host}\' is down")
                elif response.status_code == 403:
                    raise Exception(response.text())
                elif response.status_code != 200:
                    raise Exception(f"Post to API failed with status code: {response.status_code}")
                else:
                    task = response.json()
                    if task["task_id"] != -1:
                        task_id = task["task_id"]
                        miner_uids = task.get('miner_uids')
                        bt.logging.debug(f"Received organic query with task id: {task_id}")
                        if task["timeout"] == None:
                            task["timeout"] = 5.0
                        if task["chunk_size"] == None:
                            task["chunk_size"] = 4096
                        synapse = chunkSynapse(
                            document=task["document"],
                            timeout=task["timeout"],
                            chunk_size=task["chunk_size"]
                        )
                return Task(synapse=synapse, task_type="organic", task_id=task_id, miner_uids=miner_uids)
            except Exception as e:
                bt.logging.error(f"Failed to get task from API host: \'{API_host}\'. Exited with exception\n{e}")
        bt.logging.debug("Generating synthetic query")
        synapse = generate_synthetic_synapse(validator)
        return Task(synapse=synapse, task_type="synthetic", task_id=-1)

    @classmethod
    def return_response(cls, validator, response_data):
        validator_hotkey = validator.wallet.get_hotkey()
        validator_sig = sign(
            (validator_hotkey.public_key, validator_hotkey.private_key),
            str.encode(json.dumps(response_data))
            ).hex()
        API_host = os.environ['CHUNKING_API_HOST']
        task_url = f"{API_host}/task_api/organic_response/"
        headers = {"Content-Type": "application/json"}
        data = {
            'response_data': response_data,
            'validator_sig': validator_sig,
        }
        try:
            response = requests.post(task_url, headers=headers, json=data)
        except Exception as e:
            bt.logging.error(f"Failed to return response to API host: \'{API_host}\'. Exited with exception\n{e}")


    @classmethod
    def upload_logs(cls, validator, log_data):
        hotkey = validator.wallet.get_hotkey()
        signature = sign(
            (hotkey.public_key, hotkey.private_key),
            str.encode(json.dumps(log_data))
            ).hex()
            
        API_host = os.environ['CHUNKING_API_HOST']
        task_url = f"{API_host}/task_api/log/"
        headers = {"Content-Type": "application/json"}
        data = {
            'log_data': log_data,
            'signature': signature,
        }
        try:
            response = requests.post(task_url, headers=headers, json=data)
            bt.logging.debug(f"uploaded logs, response: {response.status_code}")
        except Exception as e:
            bt.logging.error(f"Failed to upload logs to API host: \'{API_host}\'. Exited with exception\n{e}")


def generate_synthetic_synapse(validator) -> chunkSynapse:
    page = choice(validator.articles)
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
    synapse = chunkSynapse(document=document, time_soft_max=5.0, chunk_size=4096)
    return synapse
