import logging
import numpy as np
import bittensor as bt
from chunking.validator.tournament import create_groups
from tests.utils.misc import compare_lists

logger = logging.getLogger(__name__)


def pretty_print_ranks(
    group_ranks: list[range],
    miner_groups: list[np.ndarray[int]] | None = None,
    with_group_index: bool = False,
):

    string = "\n"

    # num_number_skip = 1
    num_whitespace_skip = 0

    for i, group_rank_range in enumerate(group_ranks):
        row_string = ""

        whitespace_str = " " * num_whitespace_skip

        for rank in group_rank_range:
            row_string += f"{rank} "
        row_string_len = len(row_string)
        if with_group_index:
            row_string += f"({i})"

        row_string = whitespace_str + row_string

        if miner_groups:
            miner_group = miner_groups[i]
            row_string += "\n" + whitespace_str
            for miner in miner_group:
                row_string += f"{miner} "

        string += row_string + "\n"
        num_whitespace_skip += row_string_len // 2

    logger.info(string)


def test_rank_values():
    bt.debug()
    group_size = 2
    rankings = np.arange(256)
    alpha = 0.025

    miner_groups, group_ranks, group_rank_values = create_groups(rankings, group_size)
    logger.info(f"miner_groups: {miner_groups}")
    logger.info(f"group_rank_values: {group_rank_values}")
    pretty_print_ranks(group_ranks, with_group_index=True)
    logger.info(f"there are {len(miner_groups)} groups")

    # assert len(miner_groups) ==
    # assert len(group_ranks) == 4
    assert compare_lists(group_rank_values[0], [0, 1])
    assert compare_lists(group_rank_values[1], [0.5, 1.5, 2.5, 3.5])
    assert compare_lists(group_rank_values[2], [1.75, 2.75, 3.75, 4.75, 5.75, 6.75])
    assert compare_lists(
        group_rank_values[3],
        [
            4.125,
            5.125,
            6.125,
            7.125,
            8.125,
            9.125,
            10.125,
            11.125,
        ],
    )
    assert compare_lists(
        group_rank_values[4],
        [
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
        ],
    )
