from typing import List, Optional
from fastapi import HTTPException
import numpy as np
import bittensor as bt
from sr25519 import verify
import time
import asyncio
import logging

logger = logging.getLogger(__name__)

class RequestHelper:
    def __init__(self, netuid: int, network: str, whitelisted_hotkeys: Optional[List[str]] = None):
        self.task_queue = []
        self.task_status = np.empty(0)
        self.tasks = []
        self.metagraph = bt.metagraph(netuid=netuid, network=network)
        self.nonces = {}
        self.whitelist = self.initialize_whitelist(whitelisted_hotkeys)
        logger.info(f"Initialized helper with whitelist: {self.whitelist}")
        self.last_update = time.time()


    def initialize_whitelist(self, whitelisted_hotkeys: Optional[List[str]]):
        if whitelisted_hotkeys is None or len(whitelisted_hotkeys) == 0:
            return np.array(self.metagraph.hotkeys)[
                [arg[0] for arg in np.argwhere(self.metagraph.validator_trust > 0)]
            ].tolist()
        return whitelisted_hotkeys

    async def update_whitelist(self):
        while True:
            if time.time() - self.last_update > 1800:
                self.last_update = time.time()
                self.whitelist = self.initialize_whitelist(None)
            logger.info(f"Whitelist updated")
            await asyncio.sleep(10)

    def verify_signature(self, signature: str, data: str, public_key: str):
        if not verify(bytes.fromhex(signature), str.encode(data), bytes.fromhex(public_key)):
            logger.info(f"Signature mismatch")
            raise HTTPException(status_code=403, detail="Signature mismatch")

    def verify_nonce(self, hotkey: str, nonce: int):
        if self.nonces.get(hotkey) is None:
            if nonce < time.time_ns() - 4000000000:
                logger.info(f"Nonce: {nonce}, is too old. Current time: {time.time_ns()}")
                raise HTTPException(status_code=403, detail="Nonce is too old")
        elif nonce < self.nonces[hotkey]:
            logger.info(f"Nonce: {nonce}, is too old")
            raise HTTPException(status_code=403, detail="Nonce is too old")
        self.nonces[hotkey] = nonce

    def verify_task_assignment(self, task_id: int, validator_hotkey: str):
        task = self.tasks[task_id]
        if task["validator_hotkey"] != validator_hotkey:
            logger.info(f"Task with task id: {task_id} was not assigned to validator with hotkey: {validator_hotkey}")
            raise HTTPException(status_code=403, detail=f"Task with task id: {task_id} was not assigned to validator with hotkey: {validator_hotkey}")

    def verify_task_data(self, task: dict, task_data: dict):
        if (task['document'] != task_data['document'] or
            task['chunk_size'] != task_data['chunk_size'] or
            task['chunk_qty'] != task_data['chunk_qty']):
            logger.info(f"Response does not match assigned task")
            raise HTTPException(status_code=400, detail='Response does not match assigned task')

    async def send_tao(self):        
        pass


helper: RequestHelper | None = None

def init_helper(netuid: int, network: str, whitelisted_hotkeys: Optional[List[str]] = None):
    global helper
    helper = RequestHelper(netuid, network, whitelisted_hotkeys)
    helper.update_whitelist()
    
def get_helper():
    global helper
    if helper is None:
        raise Exception("Helper not initialized")
    
    return helper