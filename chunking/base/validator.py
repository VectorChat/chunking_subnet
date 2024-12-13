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

from datetime import timedelta
from functools import partial
import logging
import os
import copy
from fastapi import FastAPI, HTTPException
import httpx
import numpy as np
import asyncio
import threading
import bittensor as bt
import time
import requests

import uvicorn
import chunking
import traceback
from bittensor.core.settings import version_as_int

from dotenv import load_dotenv

from typing import List, Literal, Tuple, Union
from math import floor

from chunking.base.neuron import BaseNeuron
import wandb
from wandb.apis.public.runs import Runs, Run
import sympy as sp

from chunking.protocol import chunkSynapse
from chunking.utils.synthetic.synthetic import generate_document
from chunking.utils.synthetic.types import SyntheticGenType
from chunking.utils.wandb.wandb import WandbLogger
from chunking.validator.integrated_api import setup_routes
from chunking.validator.types import EndTournamentRoundInfo
from chunking.utils.score import get_new_scores


load_dotenv()


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

        # Save a copy of the hotkeys to local memory.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

        # Dendrite lets us send messages to other nodes (axons) in the network.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Set up initial scoring weights for validation, scores serve as the moving average of each miner's rank. A lower score is better.
        bt.logging.info("Building validation weights.")
        self.scores = np.full(
            shape=self.metagraph.n, fill_value=np.inf, dtype=np.float64
        )

        bt.logging.debug(f"Initial scores: {self.scores}")

        # initial rankings is the index of the miner in the metagraph.
        # rankings array represents rank of each miner for this validator's tournament. The index is the rank and the value is the uid of the miner.
        self.rankings = np.array(range(self.metagraph.n))

        # load tournament state from disk, if it exists.
        self.load_state()

        # Serve axon to enable external connections.
        if not self.config.neuron.axon_off:
            self.serve_axon()
        else:
            bt.logging.warning("axon off, not serving ip to chain.")

        self.app = FastAPI()
        self.score_update_queue: asyncio.Queue[EndTournamentRoundInfo] = asyncio.Queue()
        bt.logging.info("Initialized queue")
        setup_routes(self)
        bt.logging.info("Setup routes")

        # setup for running background thread
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None

        self.is_debug = self.config.debug.on
        self.allow_all_log_handlers = self.config.debug.all_log_handlers

        if self.is_debug:
            bt.logging.set_debug()
            bt.logging.set_trace()
            bt.logging.register_primary_logger("asyncio")
            if self.allow_all_log_handlers:
                bt.logging.enable_third_party_loggers()
            bt.logging.info("Debug mode enabled")

        # synthetic document queue
        self.synthetic_document_queue = asyncio.Queue[Tuple[str, int]](
            self.config.doc_gen.queue_size
        )

        self.wandb_logger = WandbLogger(self)

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
                traceback.print_exc()
                pass

        except Exception as e:
            bt.logging.error(f"Failed to create Axon initialize with exception: {e}")
            traceback.print_exc()
            pass

    def start_api(self):
        config = uvicorn.Config(
            app=self.app,
            host=self.config.task_api.host,
            port=self.config.task_api.port,
            loop="asyncio",
        )
        self.api_server = uvicorn.Server(config)

        asyncio.create_task(self.api_server.serve())

    async def concurrent_forward(self):
        """
        Run multiple forwards in parallel
        """
        coroutines = [
            self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    async def synthetic_document_producer(self):
        """
        Produce synthetic documents for the validator to query indefinitely.
        """
        bt.logging.info("Starting synthetic document producer")

        async def generate_and_add_to_queue():
            start_time = time.time()
            doc, pageid = await generate_document(self)
            end_time = time.time()
            bt.logging.info(
                f"Generated document in {end_time - start_time} seconds of length {len(doc)} chars"
            )
            await self.add_synthetic_document_to_queue(doc, pageid)

        while True:
            bt.logging.info(
                f"Generating {self.config.doc_gen.concurrent_n} synthetic documents concurrently"
            )

            await asyncio.gather(
                *[
                    generate_and_add_to_queue()
                    for _ in range(self.config.doc_gen.concurrent_n)
                ]
            )

            await asyncio.sleep(self.config.doc_gen.interval_seconds)

    async def add_synthetic_document_to_queue(
        self, synthetic_document: str, pageid: int
    ):
        """
        Add a synthetic document to the queue.
        """
        bt.logging.debug(
            f"Adding synth doc to queue of length {self.synthetic_document_queue.qsize()}, pageid: {pageid}"
        )
        await self.synthetic_document_queue.put((synthetic_document, pageid))
        bt.logging.debug(
            f"Added synth doc to queue of length {self.synthetic_document_queue.qsize()}, pageid: {pageid}"
        )

    async def get_synthetic_document_from_queue(self) -> Tuple[str, int]:
        """
        Get a synthetic document from the queue.
        """
        bt.logging.debug(
            f"Getting synth doc from queue of length {self.synthetic_document_queue.qsize()}"
        )
        doc, pageid = await self.synthetic_document_queue.get()
        bt.logging.debug(
            f"Got synth doc ({len(doc)} chars) with pageid: {pageid} from queue. New queue size: {self.synthetic_document_queue.qsize()}"
        )
        return doc, pageid

    async def run(self):
        """
        Main loop for the validator.

        This function performs the following primary tasks:
        1. Make sure the validator is registered on the network and sync the metagraph.
        2. Start main validator loop.
            2.1. Run a tournament round (currently not parallelized), in this tournament round the validator chooses a random miner group to query with a task. If organic queries are allowed,
            it checks an external task api to get an organic task (from the outside world). If organic queries are not allowed or there are no organic tasks available, it creates a synthetic task, currently
            Wikipedia articles between 10k - 100k characters. Miner are rewarded based on the chunks they return for the task. They are then ranked within their group and this new ranking is used to update the global
            moving average rank (`scores`) for each miner this moving average is then used to set the current global rankings for all miners in this validators tournament. The global ranking ultimately determines the
            weight each miner receives. The round info and updated scores/rankings are logged to wandb.
            2.2 Sync the metagraph and set weights if necessary.
            2.3 Save the current tournament state to disk.
            2.4 Sleep for a specified interval, repeat.
        """

        if self.config.enable_task_api:
            bt.logging.info("Starting integrated task API")
            self.start_api()

        # Check that validator is registered on the network.
        # self.sync()
        # self.sync_articles()
        bt.logging.info(f"Validator starting at block: {self.block}")

        await self.sync_articles()
        synth_gen_task = asyncio.create_task(self.synthetic_document_producer())

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step})")

                # TODO: consider using to_thread()

                if not self.config.wandb.wandb_off:
                    self.wandb_logger.restart_if_past_time_delta(
                        timedelta(hours=self.config.wandb.restart_interval_hours)
                    )

                # Sync metagraph and potentially set weights.
                await self.sync_articles()
                self.sync()

                if not self.config.no_forward:  # useful for debugging
                    # Run multiple forwards concurrently.
                    await self.concurrent_forward()

                # Process any queued score updates.
                await self.process_score_updates()

                # Check if we should exit.
                if self.should_exit:
                    break

                # Save the current tournament state to disk.
                await self.save_state()

                interval_seconds = self.config.neuron.synthetic_query_interval_seconds

                bt.logging.success(
                    f"step({self.step}) completed!, sleeping for {interval_seconds} seconds"
                )
                self.step += 1

                await asyncio.sleep(interval_seconds)

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            bt.logging.info("Canceling synthetic data generation task")
            synth_gen_task.cancel()
            bt.logging.success("Cancelled synthetic data generation task")
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            traceback.print_exc()

            # reinitialize subtensor just in case
            self.subtensor = bt.subtensor(config=self.config)
            # remake metagraph in case
            self.metagraph = self.subtensor.metagraph(self.config.netuid)

        finally:
            bt.logging.info("Canceling synthetic data generation task")
            synth_gen_task.cancel()
            bt.logging.success("Cancelled synthetic data generation task")

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.info("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.thread_run, daemon=True)
            self.thread.start()

            self.is_running = True
            bt.logging.success("Started")

    def thread_run(self):
        asyncio.run(self.run(), debug=self.is_debug)

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
        if self.wandb_logger:
            self.wandb_logger.finish()

        if self.is_running:
            bt.logging.debug("Stopping validator in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    @staticmethod
    def _get_raw_weights(scores: np.ndarray, rankings: np.ndarray):
        """
        Gets the raw weights based on scores/rankings.

        The top `num_weights_cap` miners receive a weight of `(1/2)^i` where `i` is the rank of the miner.
        If there are more than `num_weights_cap` active miners, the weight distribution is linear from the `num_weights_cap`th miner to the last active miner.
        """

        assert len(scores) == len(
            rankings
        ), "scores and rankings must be the same length"

        if not isinstance(scores, np.ndarray):
            bt.logging.warning(
                f"scores is not a numpy array, found {type(scores)}, converting to numpy array"
            )
            scores = np.array(scores)

            if not isinstance(rankings, np.ndarray):
                bt.logging.warning(
                    f"rankings is not a numpy array, found {type(rankings)}, converting to numpy array"
                )
                rankings = np.array(rankings)

        bt.logging.debug(f"scores len: {len(scores)}, rankings len: {len(rankings)}")

        num_weights_cap = 7

        # initialize raw weights to 0
        n = len(scores)
        raw_weights = np.zeros(n)

        # assign weights to top `num_weights_cap` miners
        i = 0
        for uid in rankings:
            if i >= num_weights_cap:
                break
            if np.isinf(scores[uid]):
                continue
            raw_weights[uid] = (1 / 2) ** i  # (1/2)^i where i is the rank (0-indexed)
            i += 1

        # num active miners is number of uids with finite scores
        num_active_miners = np.sum(np.isfinite(scores))

        bt.logging.debug(f"num_active_miners: {num_active_miners}")

        # only use linear distro if there are more than num_weights_cap active miners,
        # and we are not at the last active miner
        if (
            i >= num_weights_cap
            and i < num_active_miners
            and scores[rankings[i]] != np.inf
        ):
            # calculate the weight that would be given to place `num_weights_cap` (or the last place if fewer than `num_weights_cap`)
            last_top_weight = (1 / 2) ** (min(num_weights_cap, i))

            left = i
            right = num_active_miners

            # first constraint, integration from left to right should be equal to `last_top_weight`
            m_1 = (right**2 / 2) - (left**2 / 2)
            b_1 = right - left
            r_1 = last_top_weight

            # second constraint, y = 0 at right point
            m_2 = right
            b_2 = 1
            r_2 = 0

            matrix = np.array([[m_1, b_1, r_1], [m_2, b_2, r_2]])

            bt.logging.debug(f"matrix: {matrix}")

            # Solve for m and b
            matrix_rref = sp.Matrix(matrix).rref()
            bt.logging.debug(f"matrix_rref: {matrix_rref}")

            true_m = matrix_rref[0][2]
            true_b = matrix_rref[0][5]

            # linear function that assigns weights to miners
            def f(x: int):
                return max(0, true_m * x + true_b)

            bt.logging.debug(f"f({left}) = {f(left)}, f({right}) = {f(right)}")

            total = 0

            # assign weights to remaining miners
            for rank in range(left, min(right, n)):
                uid = rankings[rank]

                total += f(rank)
                if np.isinf(scores[uid]):
                    break
                raw_weights[uid] = f(rank)

            bt.logging.debug(
                f"last_top_weight = {last_top_weight}, linear fn sum = {total}"
            )

        return raw_weights

    def set_weights_on_chain(
        self, uint_uids: List[int], uint_weights: List[int]
    ) -> Tuple[bool, str]:
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        return result, msg

    def set_weights(self: "BaseValidatorNeuron"):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """

        bt.logging.debug("setting weights")

        if np.isnan(self.scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # bt.logging.debug(f"self.scores = {self.scores}")

        if len(self.scores) != len(self.rankings):
            bt.logging.warning(
                f"scores and rankings are different lengths, adjusting rankings to match scores"
            )
            self.rankings = np.argsort(self.scores)

        raw_weights = self._get_raw_weights(self.scores, self.rankings)

        # bt.logging.debug("raw_weights", raw_weights)
        # bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = chunking.base.utils.process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
            skip_exclude=True,
        )
        # bt.logging.debug("processed_weights", processed_weights)
        # bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.trace("uint_weights", uint_weights)
        bt.logging.trace("uint_uids", uint_uids)

        # log the weights that would be set on chain
        wandb_data = {"weights": {}}
        for uid, weight in zip(uint_uids, uint_weights):
            wandb_data["weights"][str(uid)] = weight
        if not self.config.wandb.wandb_off:
            wandb.log(wandb_data)

        if self.config.neuron.skip_set_weights_extrinsic:
            bt.logging.warning("Skipping set_weights extrinsic call.")
            return

        try:
            bt.logging.debug(f"setting weights on chain: ")
            result, msg = self.set_weights_on_chain(uint_uids, uint_weights)
            if result is True:
                bt.logging.success(
                    f"set_weights extrinsic submitted successfully!: {msg}"
                )
            else:
                bt.logging.error(f"set_weights failed: {msg}")
        except Exception as e:
            bt.logging.error(f"Error setting weights: {e}")
            traceback.print_exc()

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
        # if previous_metagraph.axons == self.metagraph.axons:
        #     bt.logging.debug("metagraph axons are the same, nothing to update")
        #     return

        bt.logging.debug("Metagraph updated, re-syncing hotkeys")

        # Reset scores for all hotkeys that have been replaced
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != self.metagraph.hotkeys[uid]:
                self.scores[uid] = np.inf

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and scores
        if len(self.hotkeys) < len(self.metagraph.hotkeys):
            cur_scores = self.scores
            placeholder_scores = np.full((self.metagraph.n), np.inf).astype(np.float64)

            placeholder_scores[: len(cur_scores)] = cur_scores

            self.scores = placeholder_scores
            bt.logging.debug(f"Added new hotkeys, new scores: {self.scores}")

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        bt.logging.debug(f"Updated hotkeys")

    async def queue_score_update(
        self,
        end_tournament_round_info: EndTournamentRoundInfo,
    ):
        try:
            bt.logging.debug(
                f"Queueing score update for {end_tournament_round_info.miner_group_uids}"
            )
            await self.score_update_queue.put(end_tournament_round_info)
        except Exception as e:
            bt.logging.error(f"Error queuing score update: {e}")
            traceback.print_exc()

    async def process_score_updates(self):
        while not self.score_update_queue.empty():
            end_tournament_round_info = await self.score_update_queue.get()
            bt.logging.debug(
                f"Processing score update for {end_tournament_round_info.miner_group_uids}, task type: {end_tournament_round_info.task_type}"
            )
            await self.update_scores(end_tournament_round_info)

    async def update_scores(
        self,
        end_tournament_round_info: EndTournamentRoundInfo,
    ):
        """
        Updates `self.scores` and `self.rankings` for the miners that were part of a specific tournament round.

        `self.scores` is the exponential moving average rank of the miners. Index is uid, value is score.
        `self.rankings` is the sorted list of uids based on the scores (rank moving average). Index is miner's global ranking in tournament, value is uid.

        Args:
            wandb_data (dict): Dictionary to store data for wandb logging.
            ranks (np.ndarray): Array of ranks for the miners, length of num miners in tournament round.
            uids (List[int]): List of uids for the miners, length of num miners in tournament round.
            task_type (Literal["organic", "synthetic"]): Type of task that the miners were part of.
            alpha (float): Exponential moving average factor.
        """

        wandb_data = end_tournament_round_info.wandb_data
        do_wandb_log = end_tournament_round_info.do_wandb_log
        uids = end_tournament_round_info.miner_group_uids

        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        self.scores = self.scores.astype(np.float64)

        self.scores = get_new_scores(
            scores=self.scores,
            uids=uids_array,
            alpha=end_tournament_round_info.alpha,
            group_best_possible_rank_value=end_tournament_round_info.group_best_possible_rank_value,
            rank_values=end_tournament_round_info.rank_values,
            miner_group_index=end_tournament_round_info.miner_group_index,
        )

        old_rankings = copy.deepcopy(self.rankings)
        self.rankings = np.argsort(self.scores)

        # print old/new rankings
        for uid in uids_array:
            bt.logging.debug(
                f"uid: {uid}. Rank: {old_rankings.tolist().index(uid)} -> {self.rankings.tolist().index(uid)}"
            )

        # log scores and rankings and other data to wandb for synthetic queries
        if do_wandb_log:
            for uid in range(len(self.scores)):
                wandb_data["all"]["scores"][str(uid)] = self.scores[uid]
            bt.logging.debug("added final scores for all to wandb log")

            for rank in range(len(self.rankings)):
                uid = self.rankings[rank]
                wandb_data["all"]["rankings"][str(uid)] = rank
            bt.logging.debug("adding final global rankings for all to wandb log")

            bt.logging.debug(
                f"Logging wandb data for {end_tournament_round_info.task_type} tournament round with uids {uids_array}"
            )
            self.wandb_log(wandb_data)

    def wandb_log(self, wandb_data: dict):
        self.wandb_logger.log(wandb_data)

    async def save_state(self):
        """Saves the state of the validator to a file."""
        bt.logging.info("Saving validator state.")

        func = partial(
            np.savez,
            file=self.config.neuron.full_path + "/state.npz",
            step=self.step,
            scores=self.scores,
            rankings=self.rankings,
            articles=self.articles,
            hotkeys=self.hotkeys,
        )

        bt.logging.debug(
            f"Async saving state to {self.config.neuron.full_path}/state.npz"
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, func)

        bt.logging.info(f"Saved validator state.")
        bt.logging.debug(
            f"Saved state for {len(self.hotkeys)} hotkeys, saved {len(self.articles)} articles"
        )

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

        bt.logging.info(
            f"Loaded validator state from {self.config.neuron.full_path}/state.npz"
        )
        # bt.logging.debug(
        #     f"Loaded state: Step: {self.step}, Scores: {self.scores}, Hotkeys: {self.hotkeys}, rankings: {self.rankings}, {len(self.articles)} articles"
        # )
        bt.logging.debug(
            f"Loaded state: Step: {self.step}, {len(self.hotkeys)} hotkeys, {len(self.articles)} articles, {len(self.rankings)} rankings ({str(self.rankings.tolist())[:40]}...), {len(self.scores)} scores ({str(self.scores.tolist())[:40]}...)"
        )

    def preprocess_synapse_for_request(
        self,
        target_axon_info: "bt.AxonInfo",
        synapse: "bt.Synapse",
        timeout: float = 12.0,
    ) -> "bt.Synapse":
        """
        Preprocesses the synapse for making a request. This includes building headers for Dendrite and Axon and signing the request.

        Args:
            target_axon_info (bittensor.core.chain_data.axon_info.AxonInfo): The target axon information.
            synapse (bittensor.core.synapse.Synapse): The synapse object to be preprocessed.
            timeout (float): The request timeout duration in seconds. Defaults to ``12.0`` seconds.

        Returns:
            bittensor.core.synapse.Synapse: The preprocessed synapse.
        """
        # Set the timeout for the synapse
        synapse.timeout = timeout
        synapse.dendrite = bt.TerminalInfo(
            ip=self.dendrite.external_ip,
            version=version_as_int,
            nonce=time.time_ns(),
            uuid=self.dendrite.uuid,
            hotkey=self.dendrite.keypair.ss58_address,
        )

        # Build the Axon headers using the target axon's details
        synapse.axon = bt.TerminalInfo(
            ip=target_axon_info.ip,
            port=target_axon_info.port,
            hotkey=target_axon_info.hotkey,
        )

        # Sign the request using the dendrite, axon info, and the synapse body hash
        message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{synapse.axon.hotkey}.{synapse.dendrite.uuid}.{synapse.body_hash}"
        synapse.dendrite.signature = f"0x{self.dendrite.keypair.sign(message).hex()}"

        return synapse

    async def query_axon(
        self, axon: bt.axon, synapse: bt.Synapse, timeout: float
    ) -> bt.Synapse:

        start_time = time.time()
        target_axon = axon.info() if isinstance(axon, bt.Axon) else axon

        url = self.dendrite._get_endpoint_url(target_axon, synapse.__class__.__name__)

        synapse = self.preprocess_synapse_for_request(
            target_axon_info=target_axon,
            synapse=synapse,
            timeout=timeout,
        )

        try:
            async with httpx.AsyncClient() as client:
                bt.logging.trace(
                    f"dendrite | --> | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | 0 | Success"
                )

                bt.legacy_encrypt_keyfile_data
                response = await client.post(
                    url,
                    headers=synapse.to_headers(),
                    json=synapse.model_dump(),
                    timeout=timeout,
                )

                json_response = response.json()

                # process the response

                if response.status_code == 200:
                    new_synapse = synapse.__class__(**json_response)
                    for key in synapse.model_dump().keys():
                        try:
                            setattr(synapse, key, getattr(new_synapse, key))
                        except Exception as e:
                            # bt.logging.error(f"Error setting attribute {key}: {e}")
                            # traceback.print_exc()
                            pass
                else:
                    if synapse.axon is None:
                        synapse.axon = bt.TerminalInfo()
                    synapse.axon.status_code = response.status_code
                    synapse.axon.status_message = json_response.get("message")

                server_headers = bt.Synapse.from_headers(response.headers)

                # Merge dendrite headers
                synapse.dendrite.__dict__.update(
                    {
                        **synapse.dendrite.model_dump(exclude_none=True),
                        **server_headers.dendrite.model_dump(exclude_none=True),
                    }
                )

                # Merge axon headers
                synapse.axon.__dict__.update(
                    {
                        **synapse.axon.model_dump(exclude_none=True),
                        **server_headers.axon.model_dump(exclude_none=True),
                    }
                )

                # Update the status code and status message of the dendrite to match the axon
                synapse.dendrite.status_code = synapse.axon.status_code  # type: ignore
                synapse.dendrite.status_message = synapse.axon.status_message  # type: ignore

                synapse.dendrite.process_time = str(time.time() - start_time)  # type: ignore

        except Exception as e:
            # bt.logging.error(f"Error querying axon: {e}")
            # bt.logging.trace(traceback.format_exc())

            synapse.dendrite.status_code = 500
            synapse.dendrite.status_message = str(e)

        finally:
            if synapse.axon is not None and synapse.dendrite is not None:
                bt.logging.trace(
                    f"dendrite | <-- | {synapse.get_total_size()} B | {synapse.name} | {synapse.axon.hotkey} | {synapse.axon.ip}:{str(synapse.axon.port)} | {synapse.dendrite.status_code} | {synapse.dendrite.status_message} | {synapse.dendrite.process_time} seconds"
                )

            return synapse

    async def query_axons(
        self, axons: list[bt.axon], synapse: bt.Synapse, timeout: float
    ) -> list[bt.Synapse]:
        if self.config.query_axons_type == "custom":
            coros = [
                self.query_axon(axon, synapse.model_copy(), timeout) for axon in axons
            ]
            responses = await asyncio.gather(*coros)
            return responses
        elif (
            self.config.query_axons_type == "bt"
            or self.config.query_axons_type == "bittensor"
        ):
            return await self.dendrite.forward(
                axons=axons,
                deserialize=False,
                synapse=synapse,
                timeout=timeout,
            )
        else:
            raise ValueError(
                f"Invalid query_axons_type: {self.config.query_axons_type}. Must be 'custom', 'bt', or 'bittensor'."
            )

    async def sync_articles(self):
        try:
            bt.logging.debug(f"syncing articles")
            articles = []

            async with httpx.AsyncClient() as client:
                res = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "format": "json",
                        "list": "categorymembers",
                        "cmpageid": "8966941",
                        "cmprop": "ids",
                        "cmlimit": "max",
                    },
                )
                response = res.json()

                articles.extend(
                    [page["pageid"] for page in response["query"]["categorymembers"]]
                )
                continuation = response.get("continue")
                while continuation is not None:
                    res = await client.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={
                            "action": "query",
                            "format": "json",
                            "list": "categorymembers",
                            "cmpageid": "8966941",
                            "cmprop": "ids",
                            "cmlimit": "max",
                            "cmcontinue": continuation.get("cmcontinue"),
                        },
                    )
                    response = res.json()
                    continuation = response.get("continue")
                    articles.extend(
                        [
                            page["pageid"]
                            for page in response["query"]["categorymembers"]
                        ]
                    )
            self.articles = articles
            bt.logging.debug(f"synced articles!")
        except Exception as e:
            bt.logging.error(f"Error syncing articles: {e}")
            traceback.print_exc()
