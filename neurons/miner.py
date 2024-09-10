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


import time
from typing import List, Tuple

import bittensor as bt

import chunking
from chunking.base.miner import BaseMinerNeuron

from nltk.tokenize import sent_tokenize
import json
from sr25519 import sign
from substrateinterface import Keypair

from bittensor.errors import (
    SynapseDendriteNoneException
)
from bittensor.constants import V_7_2_0

class Miner(BaseMinerNeuron):

    def __init__(self):
        super(Miner, self).__init__()
        
        self.nonces = {}

    async def forward(
        self, synapse: chunking.protocol.chunkSynapse
    ) -> chunking.protocol.chunkSynapse:
        """
        Processes the incoming chunkSynapse and returns response.

        Args:
            synapse (chunking.protocol.chunkSynapse): The synapse object containing the document.

        Returns:
            chunking.protocol.chunkSynapse: The synapse object with the 'chunks' field set to the generated chunks.

        """
        # default miner logic, see docs/miner_guide.md for help writing your own miner logic

        bt.logging.debug(f"from hotkey {synapse.dendrite.hotkey[:10]}: Received chunk_size: {synapse.chunk_size}, time_soft_max: {synapse.time_soft_max}")

        document = sent_tokenize(synapse.document)
        bt.logging.debug(f"From hotkey {synapse.dendrite.hotkey[:10]}: Received query: \"{document[0]} ...\"")
        
        
        chunks = []       
        while len(document) > 0:
            chunks.append(document[0])
            del document[0]
            while len(document) > 0:
                if len(chunks[-1] + " " + document[0]) > synapse.chunk_size:
                    break
                chunks[-1] += (" " + document.pop(0))

        bt.logging.debug(f"Created {len(chunks)} chunks")

        synapse.chunks = chunks

        response_data = {
            'document': synapse.document,
            'chunk_size': synapse.chunk_size,
            'chunk_qty': synapse.chunk_qty,
            'chunks': synapse.chunks,
        }
                
        synapse.miner_signature = str(sign(
            (self.wallet.get_hotkey().public_key, self.wallet.get_hotkey().private_key),
            str.encode(json.dumps(response_data))
        ).hex())
        
        bt.logging.debug(f"signed synapse with signature: {synapse.miner_signature}")
        
        return synapse

    async def blacklist(
        self, synapse: chunking.protocol.chunkSynapse
    ) -> Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (chunking.protocol.chunkSynapse): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not synapse.dendrite.hotkey or
            not synapse.dendrite.hotkey in self.metagraph.hotkeys
        ):
            if self.config.blacklist.allow_non_registered:
                bt.logging.warning(f"Accepting request from un-registered hotkey {synapse.dendrite.hotkey}")
                return False, "Allowing un-registered hotkey"
            else:
                # Ignore requests from un-registered entities.
                bt.logging.warning(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
                return True, "Unrecognized hotkey"

        if not self.metagraph.validator_permit[uid]:
            if self.config.blacklist.force_validator_permit:
                # Ignore request from non-validator
                bt.logging.warning(f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}")
                return True, "Non-validator hotkey"
            else:
                bt.logging.warning(f"Accepting request from non-validator hotkey {synapse.dendrite.hotkey}")
                return False, "Validator permit not required"
            
        stake = self.metagraph.S[uid].item()
        
        if stake < self.config.blacklist.minimum_stake:
            # Ignore request from entity with insufficient stake.
            bt.logging.warning(f"Blacklisting request from hotkey {synapse.dendrite.hotkey} with insufficient stake: {stake}")
            return True, "Insufficient stake"

        bt.logging.debug(f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, "Hotkey recognized!"

    async def priority(
        self, synapse: chunking.protocol.chunkSynapse
    ) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (chunking.protocol.chunkSynapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)  # Get the caller index.
        priority = float(self.metagraph.S[caller_uid])  # Return the stake as the priority.
        bt.logging.debug(f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority)
        return priority

    def _to_nanoseconds(self, seconds: float) -> int:
        return int(seconds * 1_000_000_000)

    def _to_seconds(self, nanoseconds: int) -> float:
        return float(nanoseconds / 1_000_000_000)

    async def verify(
        self, synapse: chunking.protocol.chunkSynapse
    ) -> None:
        
        if self.config.neuron.disable_verification:
            bt.logging.warning("Verification disabled")
            return
        
         # Build the keypair from the dendrite_hotkey
        if synapse.dendrite is not None:
            keypair = Keypair(ss58_address=synapse.dendrite.hotkey)

            # Build the signature messages.
            message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{self.wallet.hotkey.ss58_address}.{synapse.dendrite.uuid}.{synapse.computed_body_hash}"

            # Build the unique endpoint key.
            endpoint_key = f"{synapse.dendrite.hotkey}:{synapse.dendrite.uuid}"

            # Requests must have nonces to be safe from replays
            if synapse.dendrite.nonce is None:
                raise Exception("Missing Nonce")
                        
                        

            if synapse.dendrite.version is not None and synapse.dendrite.version >= V_7_2_0:
                bt.logging.debug(f"Using custom synapse verification logic")
                # If we don't have a nonce stored, ensure that the nonce falls within
                # a reasonable delta.
                cur_time = time.time_ns()
                
                allowed_delta = min(self.config.neuron.synapse_verify_allowed_delta, self._to_nanoseconds(synapse.timeout or 0))
                
                latest_allowed_nonce = synapse.dendrite.nonce + allowed_delta                                
                
                bt.logging.debug(f"synapse.dendrite.nonce: {synapse.dendrite.nonce}")
                bt.logging.debug(f"latest_allowed_nonce: {latest_allowed_nonce}")
                bt.logging.debug(f"cur time: {cur_time}")
                bt.logging.debug(f"delta: {self._to_seconds(cur_time - synapse.dendrite.nonce)}")
                
                if (
                    self.nonces.get(endpoint_key) is None
                    and synapse.dendrite.nonce > latest_allowed_nonce
                ):
                    raise Exception(f"Nonce is too old. Allowed delta in seconds: {self._to_seconds(allowed_delta)}, got delta: {self._to_seconds(cur_time - synapse.dendrite.nonce)}")
                if (
                    self.nonces.get(endpoint_key) is not None
                    and synapse.dendrite.nonce <= self.nonces[endpoint_key]
                ):
                    raise Exception(f"Nonce is too small, already have a newer nonce in the nonce store, got: {synapse.dendrite.nonce}, already have: {self.nonces[endpoint_key]}")
            else:
                bt.logging.warning(f"Using synapse verification logic for version < 7.2.0: {synapse.dendrite.version}")
                if (
                    endpoint_key in self.nonces.keys()
                    and self.nonces[endpoint_key] is not None
                    and synapse.dendrite.nonce <= self.nonces[endpoint_key]
                ):
                    raise Exception(f"Nonce is too small, already have a newer nonce in the nonce store, got: {synapse.dendrite.nonce}, already have: {self.nonces[endpoint_key]}")

            if not keypair.verify(message, synapse.dendrite.signature):
                raise Exception(
                    f"Signature mismatch with {message} and {synapse.dendrite.signature}, from hotkey {synapse.dendrite.hotkey}"
                )

            # Success
            self.nonces[endpoint_key] = synapse.dendrite.nonce  # type: ignore
        else:
            raise SynapseDendriteNoneException(synapse=synapse)

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            # bt.logging.info("Miner running...", time.time())
            time.sleep(10)
