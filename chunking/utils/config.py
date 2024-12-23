# The MIT License (MIT)
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

from ast import parse
from email.policy import default
import os
import argparse
import bittensor as bt
import chunking


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    print("full path:", full_path)
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)


def add_args(cls, parser):
    """
    Adds relevant arguments to the parser for operation.
    """
    # Netuid Arg: The netuid of the subnet to connect to.
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

    neuron_type = "validator" if "miner" not in cls.__name__.lower() else "miner"

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default=neuron_type,
    )

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default="cpu",
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=100,
    )

    parser.add_argument(
        "--neuron.sync_metagraph_interval",
        type=int,
        help="The interval between metagraph syncs in blocks.",
        default=50,
    )

    if neuron_type == "validator":

        parser.add_argument(
            "--neuron.timeout",
            type=float,
            help="The timeout for each forward call in seconds.",
            default=20,
        )

        parser.add_argument(
            "--neuron.num_concurrent_forwards",
            type=int,
            help="The number of concurrent forwards running at any time.",
            default=1,
        )

        parser.add_argument(
            "--neuron.sample_size",
            type=int,
            help="The number of miners to query in a single step.",
            default=2,
        )

        parser.add_argument(
            "--neuron.disable_set_weights",
            action="store_true",
            help="Disables setting weights.",
            default=False,
        )

        parser.add_argument(
            "--neuron.min_moving_average_alpha",
            type=float,
            help="Moving average alpha parameter used for top miners, scores of lower ranked miners are adjusted with a higher alpha.",
            default=0.025,
        )

        parser.add_argument(
            "--neuron.axon_off",
            "--axon_off",
            action="store_true",
            # Note: the validator needs to serve an Axon with their IP or they may
            #   be blacklisted by the firewall of serving peers on the network.
            help="Set this flag to not attempt to serve an Axon.",
            default=False,
        )

        parser.add_argument(
            "--num_embeddings",
            type=int,
            help="Number of embeddings to generate and compare.",
            default=150,
        )
        parser.add_argument(
            "--accept_organic_queries",
            action="store_true",
            help="Set this flag to accept organic queries",
            default=False,
        )

        parser.add_argument(
            "--neuron.skip_set_weights_extrinsic",
            action="store_true",
            help="Skip the set_weights extrinsic call (only logs to W&B).",
            default=False,
        )

        parser.add_argument(
            "--neuron.synthetic_query_interval_seconds",
            type=int,
            help="The interval between synthetic queries in seconds.",
            default=0,
        )

        parser.add_argument(
            "--wandb.project_name",
            type=str,
            help="The name of the wandb project.",
            default=chunking.PROJECT_NAME,
        )

        parser.add_argument(
            "--wandb.wandb_off",
            action="store_true",
            help="Turn off wandb logging.",
        )

        parser.add_argument(
            "--wandb.log_stdout_if_off",
            action="store_true",
            help="If set, logs to stdout if wandb is off.",
            default=False,
        )

        parser.add_argument(
            "--wandb.restart_interval_hours",
            type=float,
            help="The interval between wandb restarts in hours.",
            default=24,
        )

        parser.add_argument(
            "--neuron.use_wiki_gen",
            action="store_true",
            help="Only to be used for debugging, gets documents from wikipedia instead of generating from LLM",
        )

        parser.add_argument(
            "--enable_task_api",
            action="store_true",
            help="If set, runs the integrated API for that can be queried by external clients.",
        )

        parser.add_argument(
            "--task_api.host",
            type=str,
            help="The host for the task API.",
            default="0.0.0.0",
        )

        parser.add_argument(
            "--task_api.port",
            type=int,
            help="The port for the task API.",
            default=8080,
        )

        parser.add_argument(
            "--debug.on",
            action="store_true",
            help="If set, runs the validator in debug mode.",
            default=False,
        )

        parser.add_argument(
            "--debug.all_log_handlers",
            action="store_true",
            help="If in debug mode, allows all log handlers to be registered.",
            default=False,
        )

        parser.add_argument(
            "--doc_gen.queue_size",
            type=int,
            help="The size of the synthetic document queue.",
            default=10,
        )

        parser.add_argument(
            "--doc_gen.concurrent_n",
            type=int,
            help="The number of concurrent document generation tasks to run.",
            default=6,
        )

        parser.add_argument(
            "--doc_gen.timeout",
            type=float,
            help="Time to weight before timing out a synthetic doc generation task",
            default=160,
        )

        parser.add_argument(
            "--no_forward",
            action="store_true",
            help="If set, does not forward queries to miners. Useful for debugging.",
            default=False,
        )

        parser.add_argument(
            "--query_axons_type",
            type=str,
            help="The type of query axons to use.",
            default="custom",
        )

    # Miner
    else:
        parser.add_argument(
            "--blacklist.force_validator_permit",
            action="store_true",
            help="If set, we will force incoming requests to have a permit.",
            default=False,
        )

        parser.add_argument(
            "--blacklist.allow_non_registered",
            action="store_true",
            help="If set, miners will accept queries from non registered entities. (Dangerous!)",
            default=False,
        )

        parser.add_argument(
            "--blacklist.minimum_stake",
            type=int,
            help="If set, we will force incoming requests to have a weight settable stake.",
            default=0,
        )

        parser.add_argument(
            "--neuron.disable_verification",
            action="store_true",
            help="If set, miners will accept queries without verifying. (Dangerous!)",
            default=False,
        )

        parser.add_argument(
            "--neuron.synapse_verify_allowed_delta",
            type=int,
            help="The allowed delta for synapse verification in nanoseconds.",
            default=10_000_000_000,  # 10 seconds
        )

        parser.add_argument(
            "--neuron.relay_embed_threshold",
            type=int,
            help="The threshold of cosine similarity to use when comparing two request documents. If the similarity is greater than this threshold, we will consider this a fuzzy duplicate and not process the request.",
            default=0.9,
        )

        parser.add_argument(
            "--neuron.check_ipfs",
            action="store_true",
            help="If set, runs IPFS/relay mining related checks.",
            default=False,
        )

        parser.add_argument(
            "--neuron.no_check_duplicate_ipfs",
            action="store_true",
            help="If set, does not check for exact or fuzzy duplicate requests in IPFS.",
            default=False,
        )

        parser.add_argument(
            "--neuron.no_serve",
            action="store_true",
            help="If set, does not serve the miner axon.",
            default=False,
        )

        parser.add_argument(
            "--neuron.reconnect.min_seconds",
            type=int,
            help="The minimum number of seconds to wait before reconnecting to the network.",
            default=2,
        )

        parser.add_argument(
            "--neuron.reconnect.max_seconds",
            type=int,
            help="The maximum number of seconds to wait before reconnecting to the network.",
            default=70,
        )

        parser.add_argument(
            "--neuron.reconnect.max_attempts",
            type=int,
            help="The maximum number of attempts to reconnect to the network.",
            default=10,
        )


def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    bt.trace()
    bt.debug()
    cls.add_args(parser)
    return bt.config(parser)
