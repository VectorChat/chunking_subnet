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

import traceback
import bittensor as bt
from random import choice
from chunking.validator.task_api import Task
from chunking.validator.tournament import (
    run_tournament_round,
)
from chunking.validator.task_api import Task
from neurons.validator import Validator


async def forward(self: Validator):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Handles 2 cases:
     - Organic query coming in through API
     - Generated query when there are no queries coming in

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.
        synapse: The chunkSynapse containing the organic query
    """

    try:
        task = await Task.get_new_task(validator=self)
    except Exception as e:
        bt.logging.error(f"Error getting new synthetic task: {e}")
        bt.logging.error(traceback.format_exc())
        return

    results = await run_tournament_round(self, task=task, do_wandb_log=True)

    if len(results) == 0:
        bt.logging.error("No results from tournament round")
        return

    info = results[0]

    if info is not None:
        bt.logging.debug(
            f"Queueing score update for group {info.miner_group_index}, uids: {info.miner_group_uids}, rank values {info.ranked_responses_global}"
        )
        await self.queue_score_update(info)
