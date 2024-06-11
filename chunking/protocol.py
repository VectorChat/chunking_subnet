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

from typing import Optional, List
import bittensor as bt

class chunkSynapse(bt.Synapse):
    """
    A simple chunking synapse protocol representation which uses bt.Synapse as its base.
    This protocol enables communication between the miner and the validator.

    Attributes:
    - document: str containing plaintext to be split by the miner.
    - response: List[str] containing chunks of text from document.
    """

    # Required request input, filled by sending dendrite caller. It is a base64 encoded string.
    document: str
    maxTokensPerChunk: int = None
    timeout: Optional[float] = None

    # Optional request output, filled by recieving axon.
    chunks: Optional[str] = None


    # Used for requests sent to validators
    miner_uids: Optional[List[int]] = []


    def deserialize(self) -> str:
        return self.chunks
