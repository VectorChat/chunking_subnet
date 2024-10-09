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
from typing import Dict, List, Tuple

import bittensor as bt
from openai import AsyncOpenAI

import chunking
from chunking.base.miner import BaseMinerNeuron

from nltk.tokenize import sent_tokenize, word_tokenize
import json
from sr25519 import sign
from substrateinterface import Keypair

from bittensor.errors import SynapseDendriteNoneException
from bittensor.constants import V_7_2_0

from chunking.utils.ipfs.ipfs import get_from_ipfs, get_pinned_cids
from chunking.utils.maths import calc_cosine_similarity
from chunking.utils.signature import verify_signature
from chunking.utils.relay.relay import (
    RelayPayload,
    get_recent_relay_pins,
    get_relay_payload,
    make_embeddings,
    sha256_hash,
)


class Miner(BaseMinerNeuron):

    def __init__(self):
        super(Miner, self).__init__()

        self.nonces = {}
        self.recent_queries = []
        self.aclient = AsyncOpenAI()

    async def check_duplicate(self, req_document: str, req_cid: str) -> bool:
        """
        Checks for exact and 'fuzzy' duplicates of the incoming document.

        An exact duplicate is defined as a document with the same hash as the incoming document.

        A fuzzy duplicate is defined as a document with a cosine similarity greater than the
        provided embedding threshold (`--neuron.relay_embed_threshold`). Similarities are calculated
        pairwise for the embeddings of the incoming document and the existing document.

        Args:
            req_document (str): The incoming document to check for duplicates.
            req_cid (str): The CID of the incoming document.

        Returns:
            bool: True if a duplicate is found, False otherwise.
        """

        recent_pins = await get_recent_relay_pins()

        bt.logging.info(
            f"Checking for fuzzy duplicate in {len(recent_pins)} recent pins"
        )

        embed_threshold = self.config.neuron.relay_embed_threshold
        bt.logging.debug(
            f"Using embedding threshold: {embed_threshold} to check for fuzzy duplicates"
        )

        req_doc_hash = sha256_hash(req_document)
        for pin in recent_pins:
            if pin.cid == req_cid:
                # Skip the incoming document itself.
                continue

            pin_doc_hash = pin.payload.message.document_hash

            if req_doc_hash == pin_doc_hash:
                bt.logging.info(
                    f"Found exact duplicate document with CID: {pin.cid}, hash: {req_doc_hash}"
                )
                return True

            try:
                req_embeddings = await make_embeddings(req_document, self.aclient)
            except Exception as e:
                bt.logging.error(
                    f"Error making embeddings while checking for fuzzy duplicate: {e}"
                )
                continue

            pin_embeddings = pin.payload.message.embeddings

            min_len = min(len(req_embeddings), len(pin_embeddings))

            similarities = []

            for i in range(min_len):
                req_embedding = req_embeddings[i]
                pin_embedding = pin_embeddings[i]

                cosine_similarity = calc_cosine_similarity(req_embedding, pin_embedding)
                similarities.append(cosine_similarity)
                if cosine_similarity > embed_threshold:
                    bt.logging.info(
                        f"Found fuzzy duplicate document with CID: {pin.cid}, hash: {req_doc_hash}, cosine similarity: {cosine_similarity}, current similarities: {similarities}, threshold: {embed_threshold}"
                    )
                    return True

            bt.logging.debug(f"Similarities: {similarities}")

        bt.logging.info(
            f"No fuzzy duplicate found for request document with hash: {req_doc_hash}"
        )
        return False

    async def check_synapse(self, synapse: chunking.protocol.chunkSynapse) -> bool:
        """
        Entrypoint for performing all checks to deter relay mining synapses.

        Checks:
        1) Check if the CID is pinned in IPFS.
        2) Check if the relay payload is authentic.
            - Check if the payload body is valid json.
            - Check if the signature is valid.
        3) Check for exact and fuzzy duplicates (semantically similar but with small or obscure changes).

        Args:
            synapse (chunking.protocol.chunkSynapse): The synapse object containing the document.

        Returns:
            bool: True if the synapse is valid, False otherwise.
        """
        try:
            # Check if the CID is pinned in IPFS.
            if not synapse.CID:
                bt.logging.error("No CID found in synapse")
                return False

            # Get the relay payload from IPFS.
            try:
                relay_payload = await get_relay_payload(synapse.CID, verbose=True)
            except Exception as e:
                bt.logging.error(f"Error getting content from IPFS: {e}")
                return False

            # Get the message from the relay payload.
            message = relay_payload.message
            message_str = json.dumps(message.model_dump())

            bt.logging.debug(
                f"Got message ({len(message_str)} chars): {message_str[:200]}..."
            )

            # Check if the message is valid.
            if not message_str:
                bt.logging.error("No message found in IPFS object")
                return False

            message_hash = sha256_hash(message_str)

            signature = relay_payload.signature

            bt.logging.debug(f"Got signature: {signature}")

            if not signature:
                bt.logging.error("No signature found in IPFS object")
                return False

            validator_hotkey = synapse.dendrite.hotkey

            bt.logging.debug(f"Validator hotkey: {validator_hotkey}")

            bt.logging.debug(f"Verifying signature...")
            if not verify_signature(signature, message_hash, validator_hotkey):
                bt.logging.error("Signature mismatch")
                return False

            bt.logging.debug("Signature verified")

            bt.logging.debug("Checking for exact and fuzzy duplicates...")
            is_duplicate = await self.check_duplicate(synapse.document, synapse.CID)

            if is_duplicate:
                bt.logging.info("Found duplicate, skipping request")
                return False

            bt.logging.debug("No duplicates found")

            return True
        except Exception as e:
            bt.logging.error(f"Error checking synapse: {e}")
            return False

    def default_chunker(
        self, document: str, chunk_size: int, max_num_chunks: int
    ) -> List[str]:
        """
        Default chunker for chunking the document into chunks of the specified chunk size.

        Notes:
        - This is a very simple chunker that does not take into account the semantic meaning of the text.
        - It does not factor in the maximum number of chunks.
        """

        sentences = sent_tokenize(document)

        chunks = []
        while len(sentences) > 0:
            chunks.append(sentences[0])
            del sentences[0]
            while len(sentences) > 0:
                if len(chunks[-1] + " " + sentences[0]) > chunk_size:
                    break
                chunks[-1] += " " + sentences.pop(0)

        bt.logging.debug(f"Created {len(chunks)} chunks")

        return chunks

    def chunk_document(
        self, document: str, chunk_size: int, max_num_chunks: int
    ) -> List[str]:
        """
        Entrypoint for chunking the document into chunks of the specified chunk size.

        After making your custom implementation of a chunker, you can call it here.
        """

        return self.default_chunker(document, chunk_size, max_num_chunks)

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
        bt.logging.debug(
            f"from hotkey {synapse.dendrite.hotkey[:10]}: Received chunk_size: {synapse.chunk_size}, time_soft_max: {synapse.time_soft_max}"
        )

        # Check if the synapse is being used for relay mining.
        if not await self.check_synapse(synapse):
            bt.logging.error(
                f"synapse failed check, skipping request from hotkey {synapse.dendrite.hotkey}"
            )
            return synapse

        chunks = self.chunk_document(
            synapse.document, synapse.chunk_size, synapse.chunk_qty
        )

        synapse.chunks = chunks

        response_data = {
            "document": synapse.document,
            "chunk_size": synapse.chunk_size,
            "chunk_qty": synapse.chunk_qty,
            "chunks": synapse.chunks,
        }

        synapse.miner_signature = str(
            sign(
                (
                    self.wallet.get_hotkey().public_key,
                    self.wallet.get_hotkey().private_key,
                ),
                str.encode(json.dumps(response_data)),
            ).hex()
        )

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
            not synapse.dendrite.hotkey
            or not synapse.dendrite.hotkey in self.metagraph.hotkeys
        ):
            if self.config.blacklist.allow_non_registered:
                bt.logging.warning(
                    f"Accepting request from un-registered hotkey {synapse.dendrite.hotkey}"
                )
                return False, "Allowing un-registered hotkey"
            else:
                # Ignore requests from un-registered entities.
                bt.logging.warning(
                    f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Unrecognized hotkey"

        if not self.metagraph.validator_permit[uid]:
            if self.config.blacklist.force_validator_permit:
                # Ignore request from non-validator
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"
            else:
                bt.logging.warning(
                    f"Accepting request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return False, "Validator permit not required"

        stake = self.metagraph.S[uid].item()

        if stake < self.config.blacklist.minimum_stake:
            # Ignore request from entity with insufficient stake.
            bt.logging.warning(
                f"Blacklisting request from hotkey {synapse.dendrite.hotkey} with insufficient stake: {stake}"
            )
            return True, "Insufficient stake"

        bt.logging.debug(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: chunking.protocol.chunkSynapse) -> float:
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
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.debug(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", priority
        )
        return priority

    def _to_nanoseconds(self, seconds: float) -> int:
        return int(seconds * 1_000_000_000)

    def _to_seconds(self, nanoseconds: int) -> float:
        return float(nanoseconds / 1_000_000_000)

    async def verify(self, synapse: chunking.protocol.chunkSynapse) -> None:

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

            if (
                synapse.dendrite.version is not None
                and synapse.dendrite.version >= V_7_2_0
            ):
                bt.logging.debug(f"Using custom synapse verification logic")
                # If we don't have a nonce stored, ensure that the nonce falls within
                # a reasonable delta.
                cur_time = time.time_ns()

                allowed_delta = min(
                    self.config.neuron.synapse_verify_allowed_delta,
                    self._to_nanoseconds(synapse.timeout or 0),
                )

                latest_allowed_nonce = synapse.dendrite.nonce + allowed_delta

                bt.logging.debug(f"synapse.dendrite.nonce: {synapse.dendrite.nonce}")
                bt.logging.debug(f"latest_allowed_nonce: {latest_allowed_nonce}")
                bt.logging.debug(f"cur time: {cur_time}")
                bt.logging.debug(
                    f"delta: {self._to_seconds(cur_time - synapse.dendrite.nonce)}"
                )

                if (
                    self.nonces.get(endpoint_key) is None
                    and synapse.dendrite.nonce > latest_allowed_nonce
                ):
                    raise Exception(
                        f"Nonce is too old. Allowed delta in seconds: {self._to_seconds(allowed_delta)}, got delta: {self._to_seconds(cur_time - synapse.dendrite.nonce)}"
                    )
                if (
                    self.nonces.get(endpoint_key) is not None
                    and synapse.dendrite.nonce <= self.nonces[endpoint_key]
                ):
                    raise Exception(
                        f"Nonce is too small, already have a newer nonce in the nonce store, got: {synapse.dendrite.nonce}, already have: {self.nonces[endpoint_key]}"
                    )
            else:
                bt.logging.warning(
                    f"Using synapse verification logic for version < 7.2.0: {synapse.dendrite.version}"
                )
                if (
                    endpoint_key in self.nonces.keys()
                    and self.nonces[endpoint_key] is not None
                    and synapse.dendrite.nonce <= self.nonces[endpoint_key]
                ):
                    raise Exception(
                        f"Nonce is too small, already have a newer nonce in the nonce store, got: {synapse.dendrite.nonce}, already have: {self.nonces[endpoint_key]}"
                    )

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
