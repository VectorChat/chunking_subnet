import chunking
import numpy as np


def create_rankings_array(length: int):
    """
    Create an array of rankings.
    """
    rankings = []
    for i in range(length):
        rankings.append(i)

    rankings = np.array(rankings)

    return rankings


def get_last_uid(miner_groups):
    """
    Get the last uid in the last miner group.
    """
    last_miner_group = miner_groups[-1]

    last_uid = list(last_miner_group)[-1]

    return last_uid


def get_first_uid(miner_groups):
    """
    Get the first uid in the first miner group.
    """
    first_miner_group = miner_groups[0]

    first_uid = list(first_miner_group)[0]

    return first_uid


def all_uids_exist(miner_groups, rankings):
    """
    Check if all uids are present in the miner groups.
    """
    uid_set = set()
    for group in miner_groups:
        for uid in group:
            uid_set.add(uid)

    rankings_set = set(rankings)

    return uid_set == rankings_set


def no_uid_shows_up_more_than_k_times(miner_groups, k):
    """
    Check if no uid shows up more than k times in the miner groups.
    """
    uid_dict = {}
    for group in miner_groups:
        for uid in group:
            if uid in uid_dict:
                uid_dict[uid] += 1
            else:
                uid_dict[uid] = 1

    for uid in uid_dict:
        if uid_dict[uid] > k:
            return False

    return True


def all_groups_less_than_group_size(miner_groups: list[range], group_size):
    """
    Check if all groups are less than the group size.
    """
    for group in miner_groups:
        if len(list(group)) > group_size:
            print(f"Group: {group} is larger than group size: {group_size}")
            return False

    return True


def test_miner_groups_creation():
    """
    Test the miner groups creation function.
    """

    rankings = create_rankings_array(25)
    group_size = 2
    k = 4

    miner_groups, group_ranks, group_rank_values = chunking.validator.create_groups(
        rankings, group_size
    )

    # assert len(miner_groups) == 1
    # assert len(group_ranks) == 1
    assert get_last_uid(miner_groups) == 24
    assert get_first_uid(miner_groups) == 0
    assert all_uids_exist(miner_groups, rankings)
    assert no_uid_shows_up_more_than_k_times(miner_groups, k)
    # assert all_groups_less_than_group_size(miner_groups, group_size + 3)

    rankings = create_rankings_array(194)
    miner_groups, group_ranks, group_rank_values = chunking.validator.create_groups(
        rankings, group_size
    )

    print(miner_groups)
    print(f"last group size {len(list(miner_groups[-1]))}")

    # assert len(miner_groups) == ((len(rankings) // group_size) * 2) + 1
    assert get_last_uid(miner_groups) == 193
    assert get_first_uid(miner_groups) == 0
    assert all_uids_exist(miner_groups, rankings)
    assert no_uid_shows_up_more_than_k_times(miner_groups, k)
    # assert all_groups_less_than_group_size(miner_groups, group_size + 3)

    rankings = create_rankings_array(100)
    miner_groups, group_ranks, group_rank_values = chunking.validator.create_groups(
        rankings, group_size
    )

    assert no_uid_shows_up_more_than_k_times(miner_groups, k)
    assert all_uids_exist(miner_groups, rankings)
    assert get_last_uid(miner_groups) == 99
    assert get_first_uid(miner_groups) == 0
    # assert all_groups_less_than_group_size(miner_groups, group_size + 3)

    rankings = create_rankings_array(256)
    # group_size = 13

    miner_groups, group_ranks, group_rank_values = chunking.validator.create_groups(
        rankings, group_size
    )

    assert get_last_uid(miner_groups) == 255
    assert get_first_uid(miner_groups) == 0
    assert all_uids_exist(miner_groups, rankings)
    assert no_uid_shows_up_more_than_k_times(miner_groups, k)
    # assert all_groups_less_than_group_size(miner_groups, group_size * 2)

    rankings = create_rankings_array(193)
    # group_size = 25

    miner_groups, group_ranks, group_rank_values = chunking.validator.create_groups(
        rankings, group_size
    )

    assert get_last_uid(miner_groups) == 192
    assert get_first_uid(miner_groups) == 0
    assert all_uids_exist(miner_groups, rankings)
    assert no_uid_shows_up_more_than_k_times(miner_groups, k)
    # assert all_groups_less_than_group_size(miner_groups, group_size * 2)

    rankings = create_rankings_array(10)
    # group_size = 3

    miner_groups, group_ranks, group_rank_values = chunking.validator.create_groups(
        rankings, group_size
    )

    assert get_last_uid(miner_groups) == 9
    assert get_first_uid(miner_groups) == 0

    # group_size = 20
    miner_groups, group_ranks, group_rank_values = chunking.validator.create_groups(
        rankings, group_size
    )

    assert get_last_uid(miner_groups) == 9
    assert get_first_uid(miner_groups) == 0
