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

import time
import asyncio
import threading
import traceback

import bittensor as bt

from chunking.base.neuron import BaseNeuron


class BaseMinerNeuron(BaseNeuron):
    """
    Base class for Bittensor miners.
    """

    neuron_type: str = "MinerNeuron"

    def __init__(self):
        super().__init__(config=self.config())

        # Attach functions to axon, the axon is responsible for handling incoming requests from validators with valid synapse objects.
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        bt.logging.info(f"Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
            verify_fn=(self.verify),
        )
        bt.logging.info(f"Axon created: {self.axon}")

        self.loop = asyncio.get_event_loop()

    def reconnect(self):
        """
        Reconnects to the network by attempting to sync with the subtensor and metagraph.

        Exponential backoff is used to avoid overwhelming the network.
        """
        for i in range(self.config.neuron.reconnect.max_attempts):
            sleep_time = min(
                self.config.neuron.reconnect.max_seconds,
                self.config.neuron.reconnect.min_seconds * 2**i,
            )
            try:
                self.subtensor = bt.subtensor(config=self.config)
                self.metagraph = self.subtensor.metagraph(self.config.netuid)
                break
            except Exception as e:
                bt.logging.error(
                    f"Error reconnecting to the network (attempt {i}/{self.config.neuron.reconnect.max_attempts}): {e}"
                )
                bt.logging.error(traceback.format_exc())

                bt.logging.error(f"Sleeping for {sleep_time} seconds before retrying.")
                time.sleep(sleep_time)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Starts the miner's axon, making it active on the network.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
        During each epoch of its operation, the miner waits for new blocks on the Bittensor network, updates its
        knowledge of the network (metagraph), and sets its weights. This process ensures the miner remains active
        and up-to-date with the network's latest state.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that miner is registered on the network.
        self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        if not self.config.neuron.no_serve:
            bt.logging.info(
                f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
            )
            self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        else:
            bt.logging.warning(
                f"Not serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
            )

        # Start  starts the miner's axon, making it active on the network.
        self.axon.start()

        bt.logging.info(f"Miner starting at block: {self.block}")

        # This loop maintains the miner's operations until intentionally stopped.
        while True:
            try:
                while (
                    self.block - self.last_sync_block < self.config.neuron.epoch_length
                ):
                    # Wait before checking again.
                    time.sleep(60)

                # Sync metagraph and potentially set weights.
                self.sync()
                self.step += 1

            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                exit()

            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception as e:
                bt.logging.error(f"Error during mining: {e}")
                bt.logging.error(traceback.format_exc())
                self.reconnect()
