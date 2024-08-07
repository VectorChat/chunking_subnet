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

import enum
import os
import copy
import numpy as np
import asyncio
import threading
import bittensor as bt
import time
import requests
import concurrent.futures
import chunking

from typing import List, Union
from traceback import print_exception
from math import floor

from chunking.base.neuron import BaseNeuron
from chunking.base.utils.weight_utils import process_weights_for_netuid, convert_weights_and_uids_for_emit
import wandb
#from chunking.utils.config import add_validator_argos

class BaseValidatorNeuron(BaseNeuron):
    """
    Base class for Bittensor validators. Your validator should inherit from this class.
    """

    neuron_type: str = "ValidatorNeuron"

    # @classmethod
    # def add_args(cls, parser: argparse.ArgumentParser):
    #     super().add_args(parser)
    #     add_validator_args(cls, parser)

    def __init__(self, config=None):
        super().__init__(config=self.config())
        if not self.config.neuron.wandb_off:
            if os.environ.get("WANDB_API_KEY") is None or os.environ.get("WANDB_API_KEY") == "":
                bt.logging.error("WANDB_API_KEY environment variable must be set if neuron.wandb_off is not set")
                self.config.neuron.wandb_off = True
            else:
                try:
                    run_name = f"validator-{self.uid}-{chunking.__version__}"
                    self.config.uid = self.uid
                    self.config.hotkey = self.wallet.hotkey.ss58_address
                    self.config.run_name = run_name
                    self.config.version = chunking.__version__
                    self.config.type = "validator"

                    # Initialize the wandb run for the single project
                    run = wandb.init(
                        name=run_name,
                        project=chunking.PROJECT_NAME,
                        entity=chunking.ENTITY,
                        config=self.config,
                        dir=self.config.full_path,
                        reinit=True,
                    )

                    # Sign the run to ensure it's from the correct hotkey
                    signature = self.wallet.hotkey.sign(run.id.encode()).hex()
                    self.config.signature = signature
                    wandb.config.update(self.config, allow_val_change=True)

                    bt.logging.success(f"Started wandb run for project '{chunking.PROJECT_NAME}'")
                except Exception as e:
                    bt.logging.error(f"Error in init_wandb: {e}")
                    self.config.neuron.wandb_off = True

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.

        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation
        bt.logging.info("Building validation weights.")
        self.scores = np.full(shape=self.metagraph.n, fill_value=np.inf, dtype=np.float64)
        
        bt.logging.debug(f"Initial scores: {self.scores}")
        
        self.rankings = np.array(range(self.metagraph.n))
                        
        self.load_state()        
        # Init sync with the network. Updates the metagraph.
        self.sync_articles()
        self.sync()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        # Create asyncio event loop to manage async tasks.
        self.loop = asyncio.get_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()


    def serve_axon(self):
        """Serve axon to enable external connections."""

        bt.logging.info("serving ip to chain...")
        try:
            self.axon = bt.axon(wallet=self.wallet, config=self.config)

            try:
                self.subtensor.serve_axon(
                    netuid=self.config.netuid,
                    axon=self.axon,
                )
                bt.logging.info(
                    f"Running validator {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to serve Axon with exception: {e}")
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            pass

    async def concurrent_forward(self):
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

        This function performs the following primary tasks:
        1. Check for registration on the Bittensor network.
        2. Continuously forwards queries to the miners on the network, rewarding their responses and updating the scores accordingly.
        3. Periodically resynchronizes with the chain; updating the metagraph with the latest network state and setting weights.

        The essence of the validator's operations is in the forward function, which is called every step. The forward function is responsible for querying the network and scoring the responses.

        Note:
            - The function leverages the global configurations set during the initialization of the miner.
            - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

        Raises:
            KeyboardInterrupt: If the miner is stopped by a manual interruption.
            Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        self.sync()
        self.sync_articles()
        bt.logging.info(f"Validator starting at block: {self.block}")

        interval_seconds = self.config.neuron.synthetic_query_interval_seconds
        
        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")
                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()                                   
                self.sync_articles()                      
                self.save_state()                      
                
                bt.logging.debug(f"step({self.step}) block({self.block}) completed!, sleeping for {interval_seconds} seconds")
                self.step += 1
                
                time.sleep(interval_seconds)

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(str(print_exception(type(err), err, err.__traceback__)))

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.debug("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the validator's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the validator's background operations upon exiting the context.
        This method facilitates the use of the validator in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def set_weights(self: "BaseValidatorNeuron"):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        bt.logging.debug("setting weights")
    
        # Check if self.scores contains any NaN values and log a warning if it does.
        if np.isnan(self.scores).any():
            bt.logging.warning(f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions.")


        # Calculate the average reward for each uid across non-zero values.        
        bt.logging.debug(f"self.scores = {self.scores}")
        
        
        # Calculate weights
        n = len(self.scores)
        raw_weights = np.zeros(n)    
        i = 0    
        for uid in self.rankings:
            if np.isinf(self.scores[uid]):
                continue
            raw_weights[uid] = (1/2) ** i  # (1/2)^i where i is the rank (0-indexed)            
            i += 1
            
        # min_index = raw_weights.argmin()
        # raw_weights[min_index] = -1 * (2 ** (len(self.scores) - 1)) / ((2 ** len(self.scores)) - 1)
        # bt.logging.debug(f"weight at min_index ({min_index}): {raw_weights[min_index]}")
        # next_best_weight = raw_weights.min()
        # bt.logging.debug(f"initial next_best_weight: f{next_best_weight}")
        # while raw_weights.max() > 0:
        #     next_best_weight /= 2
        #     raw_weights[raw_weights.argmin()] = next_best_weight
        
        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        timeout_seconds = self.config.set_weights_timeout_seconds

        if not self.config.neuron.wandb_off:
            wandb_data = {"weights": {}}
            for uid, weight in zip(uint_uids, uint_weights):
                wandb_data["weights"][str(uid)] = weight
            wandb.log(wandb_data)

        # Set the weights on chain via our subtensor connection.
        def set_weights_on_chain():
            result, msg = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_finalization=True,
                wait_for_inclusion=True,
                version_key=self.spec_version,
            )
            return result, msg

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(set_weights_on_chain)
            try:
                result, msg = future.result(timeout=timeout_seconds)
                if result is True:
                    bt.logging.success("set_weights on chain successfully!")
                else:
                    bt.logging.error("set_weights failed", msg)
            except concurrent.futures.TimeoutError:
                bt.logging.error(f"set_weights operation timed out after {timeout_seconds} seconds")

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")

        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        success = super().resync_metagraph()
        
        if not success:
            bt.logging.error("Metagraph sync failed, skipping this step.")
            return

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            bt.logging.debug("metagraph axons are the same, nothing to update")
            return

        bt.logging.info("Metagraph updated, re-syncing hotkeys")
        
        # Reset scores for all hotkeys that have been replaced
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:                
                self.scores[uid] = np.inf

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and scores
        if len(self.hotkeys) < len(self.metagraph.hotkeys):            
            cur_scores = self.scores                               
            placeholder_scores = np.full((self.metagraph.n), np.inf).astype(np.float64)            
            
            placeholder_scores[:len(cur_scores)] = cur_scores
            
            self.scores = placeholder_scores          
            bt.logging.debug(f"Added new hotkeys, new scores: {self.scores}")              

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    def update_scores(self, wandb_data: dict, ranks: np.ndarray, uids: List[int]):
        """Performs exponential moving average on the scores based on the rewards received from the miners."""
        self.scores = self.scores.astype(np.float64)
        # Check if rewards contains NaN values.
        if np.isnan(ranks).any():
            bt.logging.warning(f"NaN values detected in rewards: {ranks}")
            # Replace any NaN values in rewards with inf.
            ranks = np.nan_to_num(ranks, nan=np.inf)

        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Update scores with rewards produced by this step.
        alpha: float = self.config.neuron.moving_average_alpha

        bt.logging.debug(f"Previous scores: {self.scores}, ranks: {ranks}, uids: {uids_array}")            
        
        for rank, uid in zip(ranks, uids_array):
            if np.isinf(rank):
                continue

            # initialize score if it is np.inf
            if np.isinf(self.scores[uid]):
                self.scores[uid] = alpha * rank + (1 - alpha) * floor(np.sum(np.isfinite(self.scores)) / 2)
            elif self.scores[uid] < 0:
                self.scores[uid] = np.inf
            else:            
                self.scores[uid] = alpha * rank + (1 - alpha) * self.scores[uid]                

        bt.logging.debug(f"Updated moving avg scores: {self.scores}")                
        
        self.rankings = np.argsort(self.scores)

        if not self.config.neuron.wandb_off:
            for uid in uids_array:
                wandb_data["scores"][str(uid)] = self.scores[uid]
                wandb_data["rankings"][str(uid)] = list(self.rankings).index(uid)
            bt.logging.info(f"Logging wandb_data: {wandb_data}")
            wandb.log(wandb_data)        
                        
        bt.logging.debug(f"Updated rankings: {self.rankings}")

    def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        # Save the state of the validator to file.
        np.savez(
            self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            rankings=self.rankings,
            articles=self.articles,
            hotkeys=self.hotkeys,
        )
        
        bt.logging.info(f"Saved validator state.")
        bt.logging.debug(f"Saved state for {len(self.hotkeys)} hotkeys, saved {len(self.articles)} articles")

    
    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")
        if not os.path.exists(self.config.neuron.full_path + "/state.npz"):
            return

        # Load the state of the validator from file.
        state = np.load(self.config.neuron.full_path + "/state.npz")
        self.step = state["step"]
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
        self.rankings = state["rankings"]
        self.articles = state["articles"]
        
        bt.logging.info(f"Loaded validator state.")
        bt.logging.debug(f"Loaded state: Step: {self.step}, Scores: {self.scores}, Hotkeys: {self.hotkeys}, rankings: {self.rankings}, {len(self.articles)} articles")

    def sync_articles(self):
        try: 
            bt.logging.debug(f"syncing articles")
            articles = []
            response = requests.get('https://en.wikipedia.org/w/api.php', params={
                'action': 'query', 
                'format': 'json', 
                'list': 'categorymembers',
                'cmpageid': '8966941', 
                'cmprop': 'ids', 
                'cmlimit': 'max'
                }).json()
            
            articles.extend([page['pageid'] for page in response['query']['categorymembers']])
            continuation = response.get('continue')
            while continuation is not None:
                response = requests.get('https://en.wikipedia.org/w/api.php', params={
                    'action': 'query', 
                    'format': 'json', 
                    'list': 'categorymembers',
                    'cmpageid': '8966941', 
                    'cmprop': 'ids', 
                    'cmlimit': 'max',
                    'cmcontinue': continuation.get('cmcontinue')
                    }).json()                
                continuation = response.get('continue')
                articles.extend([page['pageid'] for page in response['query']['categorymembers']])        
            self.articles = articles
            bt.logging.debug(f"synced articles!")
        except Exception as e:
            bt.logging.error(f"Error syncing articles: {e}")