# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 Vector Chat

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


import os
import time

import bittensor as bt

from chunking.validator import forward

# import base validator class which takes care of most of the boilerplate
from chunking.base.validator import BaseValidatorNeuron
from openai import AsyncOpenAI, OpenAI


class Validator(BaseValidatorNeuron):

    def __init__(self):
        super(Validator, self).__init__()
        bt.logging.info("load_state()")
        self.load_state()

        if not os.environ.get("OPENAI_API_KEY"):
            raise Exception("OPENAI_API_KEY environment variable must be set.")

        if self.config.accept_organic_queries:
            os.environ["ALLOW_ORGANIC_CHUNKING_QUERIES"] = str(
                self.config.accept_organic_queries
            )

        if not os.environ.get("ALLOW_ORGANIC_CHUNKING_QUERIES"):
            os.environ["ALLOW_ORGANIC_CHUNKING_QUERIES"] = "False"

        if not os.environ.get("CHUNKING_API_HOST"):
            bt.logging.warning(
                "CHUNKING_API_HOST variable not set; defaulting to https://chunking.com/web3/api"
            )
            os.environ["CHUNKING_API_HOST"] = "https://chunking.com/web3/api/"

        self.client: OpenAI = OpenAI()
        self.aclient = AsyncOpenAI()
        self.embedding_model = "text-embedding-ada-002"
        self.num_embeddings = int(self.config.num_embeddings)
        self.sample_size = int(self.config.neuron.sample_size)

        self.check_nltk_download()

    def check_nltk_download(self):
        try:
            from nltk.tokenize import sent_tokenize, wordpunct_tokenize

            test_str = "Hello, world!"
            sent_tokenize(test_str)
            wordpunct_tokenize(test_str)
        except LookupError:
            import nltk

            nltk.download("punkt")
            nltk.download("punkt_tab")
            print("nltk downloaded")

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """

        return await forward(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            # bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(20)
