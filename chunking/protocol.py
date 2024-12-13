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

from typing import Literal, Optional, List
import bittensor as bt


chunkSynapseType = Literal["synthetic", "organic"]


class chunkSynapse(bt.Synapse):
    """
    A simple chunking synapse protocol representation which uses bt.Synapse as its base.
    This protocol enables communication between the miner and the validator.

    Attributes:
    - document: str containing plaintext to be split by the miner.
    - chunk_size: int containing the soft max characters per chunk before a penalty is applied.
    - chunk_qty: int containing the soft max number of chunks before a penalty is applied.
    - time_soft_max: float containing the maximum time the miner can take before being penalized.
    - chunks: List[str] containing chunks of text from document.
    - miner_signature: str containing the miner's signature of a json object containing document, chunk_size, chunk_qty, and chunks.

    Optional Attributes:
    - CID: str containing the IPFS CID of the the special relay mining payload

    Response Attributes:
    - chunks: List[str] containing chunks of text from document, created by the miner
    - miner_signature: str containing the miner's signature of a json object containing document, chunk_size, chunk_qty, and chunks.
    """

    name: str = "chunkSynapse"

    # Required request input
    document: str
    chunk_size: int
    chunk_qty: int
    time_soft_max: float
    timeout: float = 20.0

    # Optional request input
    CID: Optional[str] = None

    # Optional request output, filled by recieving axon.
    chunks: Optional[List[str]] = None
    miner_signature: Optional[str] = None

    def deserialize(self) -> List[str]:
        return self.chunks
