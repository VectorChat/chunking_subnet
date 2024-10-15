import logging
import numpy as np
import bittensor as bt
from chunking.validator.forward import create_groups, get_miner_groups
from tests.utils.misc import compare_lists

logger = logging.getLogger(__name__)


def test_rank_values():
    bt.debug()
    group_size = 2
    rankings = np.arange(256)
    alpha = 0.025

    miner_groups, group_ranks, group_rank_values = create_groups(rankings, group_size)
    logger.info(f"miner_groups: {miner_groups}")
    logger.info(f"group_ranks: {group_ranks}")
    logger.info(f"group_rank_values: {group_rank_values}")

    # assert len(miner_groups) == 
    # assert len(group_ranks) == 4
    assert compare_lists(group_rank_values[0], [0, 1])
    assert compare_lists(group_rank_values[1], [0.5, 1.5, 2.5, 3.5])
    assert compare_lists(group_rank_values[2], [1.75, 2.75, 3.75, 4.75, 5.75, 6.75])
    assert compare_lists(group_rank_values[3], [
        4.125,
        5.125,
        6.125,
        7.125,
        8.125,
        9.125,
        10.125,
        11.125,
    ])
    assert compare_lists(group_rank_values[4], [
        7.4375,
        8.4375,
        9.4375,
        10.4375,
        11.4375,
        12.4375,
        13.4375,
        14.4375,
        15.4375,
        16.4375,
    ])
        
