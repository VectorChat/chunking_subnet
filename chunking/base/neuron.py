# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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

import asyncio
import copy
from functools import partial
import traceback
from packaging import version as packaging_version

import bittensor as bt

from abc import ABC, abstractmethod


# Sync calls set weights and also resyncs the metagraph.
from chunking.utils.config import check_config, add_args, config
from chunking.utils.misc import ttl_get_block
from chunking import __spec_version__ as spec_version


class BaseNeuron(ABC):
    """
    Base class for Bittensor miners. This class is abstract and should be inherited by a subclass. It contains the core logic for all neurons; validators and miners.

    In addition to creating a wallet, subtensor, and metagraph, this class also handles the synchronization of the network state via a basic checkpointing mechanism based on epoch length.
    """

    neuron_type: str = "BaseNeuron"

    @classmethod
    def check_config(cls, config: "bt.Config"):
        check_config(cls, config)

    @classmethod
    def add_args(cls, parser):
        add_args(cls, parser)

    @classmethod
    def config(cls):
        return config(cls)

    subtensor: "bt.subtensor"
    wallet: "bt.wallet"
    metagraph: "bt.metagraph"
    spec_version: int = spec_version

    @property
    def block(self):
        return ttl_get_block(self)

    def __init__(self, config=None):
        base_config = copy.deepcopy(config or BaseNeuron.config())
        self.config = self.config()
        self.config.merge(base_config)
        self.check_config(self.config)

        bittensor_version = bt.__version__

        # this is for handling setting up logging configuration for older bittensor versions.
        if packaging_version.parse(bittensor_version) < packaging_version.parse(
            "7.1.2"
        ):
            logging_config = self.config.logging
            # manually set logging config
            if logging_config.logging_dir and logging_config.record_log:
                bt.logging.warning(
                    "Ignoring logging_dir and record_log in config. Logging to stdout."
                )
            if logging_config.trace:
                self.enable_trace()
            elif logging_config.debug:
                self.enable_debug()
        else:

            bt.logging.set_config(config=self.config.logging)

        # No GPU is necessary for chunking, though miners can always make a custom impl to use a gpu if they'd like.
        self.device = "cpu"  # self.config.neuron.device

        # Log the configuration for reference.
        bt.logging.info(self.config)

        # Set the last sync block to 1 to ensure the metagraph is synced on initialization.
        self.last_sync_block = 1

        # Setup bittensor objects for interacting with subtensor
        bt.logging.info("Setting up bittensor objects.")

        # The wallet holds the cryptographic key pairs for the miner.
        self.wallet = bt.wallet(config=self.config)

        # class that helps with connecting to subtensor chain, handles calling extrinsics, querying state, etc.
        self.subtensor = bt.subtensor(config=self.config)

        # class that helps with getting structured information from the subtensor chain
        self.metagraph = self.subtensor.metagraph(self.config.netuid)

        bt.logging.info(f"Wallet: {self.wallet}")
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Check if the miner is registered on the Bittensor network before proceeding further.
        self.check_registered()

        # Each miner gets a unique identity (UID) in the network for differentiation.
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(
            f"Running neuron on subnet: {self.config.netuid} with uid {self.uid} using network: {self.subtensor.chain_endpoint}"
        )
        self.step = 0

    @abstractmethod
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse: ...

    @abstractmethod
    def run(self): ...

    def sync(self):
        """
        Wrapper for synchronizing the state of the network for the given miner or validator.
        """
        self.check_registered()
        bt.logging.success("still registered")

        if self.should_sync_metagraph():
            self.resync_metagraph()

        if self.should_set_weights():
            self.set_weights()

    def check_registered(self):
        is_registered = self.subtensor.is_hotkey_registered(
            netuid=self.config.netuid,
            hotkey_ss58=self.wallet.hotkey.ss58_address,
        )
        if not is_registered:
            bt.logging.error(
                f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            exit()
        bt.logging.debug(
            f"Wallet: {self.wallet} is registered on netuid {self.config.netuid}."
        )

    def resync_metagraph(self) -> bool:
        """
        Resync the metagraph with the network. Returns True if successful, False otherwise.
        """
        bt.logging.info("resync_metagraph(), neuron: ", self.neuron_type)

        # Sync the metagraph.
        try:
            self.metagraph.sync(subtensor=self.subtensor)
        except Exception as e:
            bt.logging.error(f"Failed to sync metagraph with error: {e}")
            return False

        self.last_sync_block = self.block

        bt.logging.info("metagraph synced!")
        return True

    def should_sync_metagraph(self):

        diff = self.block - self.last_sync_block

        interval = self.config.neuron.sync_metagraph_interval

        bt.logging.debug(
            f"Block: {self.block}, Last sync: {self.last_sync_block}, Diff: {diff}, Interval: {interval} blocks"
        )

        should_sync = diff > interval

        bt.logging.debug(f"BaseNeuron: Should sync metagraph: {should_sync}")

        return should_sync

    def should_set_weights(self) -> bool:
        if self.neuron_type == "MinerNeuron":
            return False

        # Don't set weights on initialization.
        if self.step < 2:
            return False

        if self.config.neuron.disable_set_weights:
            return False

        updated = self.block - self.metagraph.last_update[self.uid]

        bt.logging.debug(
            f"Block: {self.block}, Last update: {self.metagraph.last_update[self.uid]}, Diff: {updated}"
        )

        should_set = updated > self.config.neuron.epoch_length

        bt.logging.debug(f"Should set weights: {should_set}")

        if should_set:
            bt.logging.debug(
                f"Setting weights. Diff: {updated}, Epoch length: {self.config.neuron.epoch_length}"
            )
            return True
        else:
            bt.logging.debug(
                f"Not setting weights. Diff: {updated}, Epoch length: {self.config.neuron.epoch_length}"
            )
            return False

    def save_state(self):
        pass
        # bt.logging.warning(
        #    "save_state() not implemented for this neuron. You can implement this function to save model checkpoints or other useful data."
        # )

    def load_state(self):
        pass
        # bt.logging.warning(
        #    "load_state() not implemented for this neuron. You can implement this function to load model checkpoints or other useful data."
        # )
