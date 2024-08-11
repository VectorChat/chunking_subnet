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
    uids = []
    for group in miner_groups:
        for uid in group:
            uids.append(uid)
            
    uids = np.array(uids)
    
    return np.array_equal(np.sort(uids), np.sort(rankings))

def no_uid_shows_three_times(miner_groups):
    """
    Check if no uid shows up three times in the miner groups.
    """
    uids = []
    for group in miner_groups:
        for uid in group:
            uids.append(uid)
            
    uids = np.array(uids)
    
    unique, counts = np.unique(uids, return_counts=True)
    
    return not 3 in counts

def test_miner_groups_creation():
    """
    Test the miner groups creation function.
    """        
    
    rankings = create_rankings_array(25)
    group_size = 25
    
    miner_groups, group_ranks, group_size = chunking.validator.create_groups(rankings, 25)
    
    assert len(miner_groups) == 1
    assert len(group_ranks) == 1
    assert get_last_uid(miner_groups) == 24
    assert get_first_uid(miner_groups) == 0
    assert all_uids_exist(miner_groups, rankings)
    
    rankings = create_rankings_array(194)
    miner_groups, group_ranks, group_size = chunking.validator.create_groups(rankings, group_size)
    
    assert len(miner_groups) == ((len(rankings) // group_size) * 2) + 1
    assert get_last_uid(miner_groups) == 193
    assert get_first_uid(miner_groups) == 0
    assert all_uids_exist(miner_groups, rankings)