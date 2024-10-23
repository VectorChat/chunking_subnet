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

from functools import partial
import os
import copy
from fastapi import FastAPI, HTTPException
import numpy as np
import asyncio
import threading
import bittensor as bt
import time
import requests

import uvicorn
import chunking
import traceback

from dotenv import load_dotenv

from typing import List, Literal, Union
from math import floor

from chunking.base.neuron import BaseNeuron
import wandb
from wandb.apis.public.runs import Runs, Run
import sympy as sp

from chunking.validator.integrated_api import setup_routes
from chunking.validator.types import EndTournamentRoundInfo
from chunking.utils.score import get_rank_value_to_adjusted_alpha


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

        bt.logging.info(f"wandb off: {self.config.wandb.wandb_off}")

        if not self.config.wandb.wandb_off:
            # connect to wandb run
            self._setup_wandb()
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

        self.app = FastAPI()
        self.score_update_queue: asyncio.Queue[EndTournamentRoundInfo] = asyncio.Queue()
        bt.logging.info("Initialized queue")
        setup_routes(self)
        bt.logging.info("Setup routes")

        # setup for running background thread
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: Union[threading.Thread, None] = None
        self.lock = asyncio.Lock()

    def _find_valid_wandb_run(self, runs: Runs) -> Run | None:
        """
        Find a valid wandb run for this validator given a list of wandb runs. The run must be signed by the validator's hotkey.
        """
        for run in runs:
            sig = run.config.get("signature")
            if not sig:
                continue

            verified = self.wallet.hotkey.verify(run.id.encode(), bytes.fromhex(sig))

            if verified:
                bt.logging.info(f"Found valid run: {run.id}")
                return run
            else:
                bt.logging.warning(
                    f"Found invalid run: {run.id}, looking for another run"
                )

        return None

    def _get_latest_wandb_run(self, project_name: str) -> Run | None:
        """
        Get the latest valid wandb run for this validator.
        """
        api = wandb.Api()
        latest_runs: Runs = api.runs(
            f"{chunking.ENTITY}/{project_name}",
            {
                "config.hotkey": self.wallet.hotkey.ss58_address,
                "config.type": "validator",
            },
            order="-created_at",
        )
        return self._find_valid_wandb_run(latest_runs)

    def _start_new_wandb_run(self, project_name: str, run_name: str) -> Run:
        """
        Start a new wandb run for this validator if no valid wandb run exists for this validator and version.
        """
        run = wandb.init(
            name=run_name,
            project=project_name,
            entity=chunking.ENTITY,
            config=self.config,
            dir=self.config.full_path,
            reinit=False,
        )
        signature = self.wallet.hotkey.sign(run.id.encode()).hex()
        self.config.signature = signature
        bt.logging.success(
            f"Started wandb run for project '{project_name}', name: '{run_name}', id: '{run.id}'"
        )
        return run

    def _find_existing_wandb_run(self, version: str, uid: int) -> Run | None:
        """
        Find a valid wandb run for this validator given a list of wandb runs. The run must be signed by the validator's hotkey and match the validator's uid and version.
        """
        api = wandb.Api()
        latest_runs: Runs = api.runs(
            f"{chunking.ENTITY}/{self.config.wandb.project_name}",
            {
                "config.hotkey": self.wallet.hotkey.ss58_address,
                "config.type": "validator",
                "config.version": version,
                "config.uid": uid,
            },
            order="-created_at",
        )
        bt.logging.debug(
            f"Found {len(latest_runs)} runs with version {version} and uid {uid}"
        )
        return self._find_valid_wandb_run(latest_runs)

    def _resume_wandb_run(self, run: Run, project_name: str):
        """
        Resume a wandb run for this validator.
        """
        wandb.init(
            entity=chunking.ENTITY, project=project_name, id=run.id, resume="must"
        )
        bt.logging.success(
            f"Resumed wandb run '{run.name}' for project '{project_name}'"
        )

    def _get_wandb_project_name(self):
        if self.config.subtensor.chain_endpoint == "test":
            return "chunking-testnet"
        return self.config.wandb.project_name

    def _setup_wandb(self):
        """
        Setup wandb for this validator.

        This function will start a new wandb run if no valid wandb run exists for this validator and version.
        If a valid wandb run exists, it will resume the wandb run.
        """
        if (
            os.environ.get("WANDB_API_KEY") is None
            or os.environ.get("WANDB_API_KEY") == ""
        ):
            raise Exception("WANDB_API_KEY environment variable must be set")

        else:
            try:
                project_name = self._get_wandb_project_name()

                latest_run = self._get_latest_wandb_run(project_name)

                run_name = f"validator-{self.uid}-{chunking.__version__}"

                self.config.uid = self.uid
                self.config.hotkey = self.wallet.hotkey.ss58_address
                self.config.run_name = run_name
                self.config.version = chunking.__version__
                self.config.type = "validator"

                if not latest_run:
                    self._start_new_wandb_run(project_name, run_name)
                else:
                    # check if uid or version has changed
                    if (
                        latest_run.config.get("version") != chunking.__version__
                        or latest_run.config.get("uid") != self.uid
                    ):
                        bt.logging.info(
                            f"Found run with different version or uid ({latest_run.name})"
                        )

                        existing_run = self._find_existing_wandb_run(
                            chunking.__version__, self.uid
                        )

                        if not existing_run:
                            bt.logging.info(
                                f"Could not find existing run with version {chunking.__version__} and uid {self.uid}, starting new run"
                            )
                            self._start_new_wandb_run(project_name, run_name)
                        else:
                            bt.logging.info(
                                f"Found existing run with version {chunking.__version__} and uid {self.uid}, resuming run"
                            )
                            self._resume_wandb_run(existing_run, project_name)
                    else:
                        self._resume_wandb_run(latest_run, project_name)

                # always update config
                wandb.config.update(self.config, allow_val_change=True)

            except Exception as e:
                raise Exception(f"Error in init_wandb: {e}")

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
        config = uvicorn.Config(app=self.app, host="0.0.0.0", port=8080, loop="asyncio")
        self.api_server = uvicorn.Server(config)

        self.loop.create_task(self.api_server.serve())

    async def concurrent_forward(self):
        """
        Run multiple forwards in parallel
        """
        coroutines = [
            self.forward() for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    def run(self):
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

        <<<<<<< HEAD

        =======
        >>>>>>> main
                Note:
                    - The function leverages the global configurations set during the initialization of the miner.
                    - The miner's axon serves as its interface to the Bittensor network, handling incoming and outgoing requests.

                Raises:
                    KeyboardInterrupt: If the miner is stopped by a manual interruption.
                    Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
        """

        # Check that validator is registered on the network.
        # self.sync()
        # self.sync_articles()
        bt.logging.info(f"Validator starting at block: {self.block}")

        interval_seconds = self.config.neuron.synthetic_query_interval_seconds

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Sync metagraph and potentially set weights.
                self.sync()
                self.sync_articles()

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Process any queued score updates.
                self.loop.run_until_complete(self.process_score_updates())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Save the current tournament state to disk.
                self.save_state()

                # bt.logging.debug(
                #     f"step({self.step}) block({self.block}) completed!, sleeping for {interval_seconds} seconds"
                # )
                self.step += 1

                # time.sleep(interval_seconds)

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            traceback.print_exc()

    def run_in_background_thread(self):
        """
        Starts the validator's operations in a background thread upon entering the context.
        This method facilitates the use of the validator in a 'with' statement.
        """
        if not self.is_running:
            bt.logging.info("Starting validator in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            if self.config.neuron.run_task_api:
                bt.logging.info("Starting integrated task API in background thread.")
                self.start_api()
            self.is_running = True
            bt.logging.success("Started")

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
        wandb.finish()

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

    def set_weights_on_chain(self, uint_uids: List[int], uint_weights: List[int]):
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=True,
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

        bt.logging.debug(f"self.scores = {self.scores}")

        if len(self.scores) != len(self.rankings):
            bt.logging.warning(
                f"scores and rankings are different lengths, adjusting rankings to match scores"
            )
            self.rankings = np.argsort(self.scores)

        raw_weights = self._get_raw_weights(self.scores, self.rankings)

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
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

            placeholder_scores[: len(cur_scores)] = cur_scores

            self.scores = placeholder_scores
            bt.logging.debug(f"Added new hotkeys, new scores: {self.scores}")

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)

    async def queue_score_update(
        self,
        end_tournament_round_info: EndTournamentRoundInfo,
    ):
        try:
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
        ranks = end_tournament_round_info.ranked_responses_global
        uids = end_tournament_round_info.miner_group_uids
        task_type = end_tournament_round_info.task_type
        alpha = end_tournament_round_info.alpha

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
        # bt.logging.debug(
        #     f"Previous scores: {self.scores}, ranks: {ranks}, uids: {uids_array}"
        # )

        bt.logging.debug(f"group alpha: {alpha}")

        rank_value_to_adjusted_alpha = get_rank_value_to_adjusted_alpha(ranks, alpha)

        for rank, uid in zip(ranks, uids_array):
            if np.isinf(rank):
                continue

            adjusted_alpha = rank_value_to_adjusted_alpha[rank]

            bt.logging.debug(
                f"uid: {uid}, rank: {rank}, adjusted_alpha: {adjusted_alpha}"
            )
            score_str = f"score: {self.scores[uid]} -> "

            # initialize score if it is np.inf
            if np.isinf(self.scores[uid]):
                self.scores[uid] = adjusted_alpha * rank + (1 - adjusted_alpha) * floor(
                    np.sum(np.isfinite(self.scores)) / 2
                )
            elif self.scores[uid] < 0:
                self.scores[uid] = np.inf
            else:
                self.scores[uid] = (
                    adjusted_alpha * rank + (1 - adjusted_alpha) * self.scores[uid]
                )

            score_str += f"{self.scores[uid]}"
            bt.logging.debug(score_str)

        # bt.logging.debug(f"Updated moving avg scores: {self.scores}")

        self.rankings = np.argsort(self.scores)

        # log scores and rankings and other data to wandb for synthetic queries
        if task_type == "synthetic":
            for uid in uids_array:
                # wandb_data["all_rankings"][str(uid)] = list(self.rankings).index(uid)
                wandb_data["group"]["scores"][str(uid)] = self.scores[uid]

            for uid in range(len(self.scores)):
                wandb_data["all"]["scores"][str(uid)] = self.scores[uid]

            for rank in range(len(self.rankings)):
                uid = self.rankings[rank]
                wandb_data["all"]["rankings"][str(uid)] = rank

            # bt.logging.debug(f"Logging wandb_data: {wandb_data}")
            bt.logging.info("Logging wandb_data")
            if not self.config.wandb.wandb_off:
                wandb.log(wandb_data)

        # bt.logging.debug(f"Updated rankings: {self.rankings}")

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

        bt.logging.info(f"Loaded validator state.")
        bt.logging.debug(
            f"Loaded state: Step: {self.step}, Scores: {self.scores}, Hotkeys: {self.hotkeys}, rankings: {self.rankings}, {len(self.articles)} articles"
        )

    async def query_axons(
        self, axons: list[bt.axon], synapse: bt.Synapse, timeout: float
    ):
        loop = self.loop
        func = partial(
            self.dendrite.query,
            axons=axons,
            timeout=timeout,
            synapse=synapse,
            deserialize=False,
        )
        responses: list[bt.Synapse] = await loop.run_in_executor(None, func)
        return responses

    def sync_articles(self):
        try:
            bt.logging.debug(f"syncing articles")
            articles = []
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "list": "categorymembers",
                    "cmpageid": "8966941",
                    "cmprop": "ids",
                    "cmlimit": "max",
                },
            ).json()

            articles.extend(
                [page["pageid"] for page in response["query"]["categorymembers"]]
            )
            continuation = response.get("continue")
            while continuation is not None:
                response = requests.get(
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
                ).json()
                continuation = response.get("continue")
                articles.extend(
                    [page["pageid"] for page in response["query"]["categorymembers"]]
                )
            self.articles = articles
            bt.logging.debug(f"synced articles!")
        except Exception as e:
            bt.logging.error(f"Error syncing articles: {e}")
            traceback.print_exc()
