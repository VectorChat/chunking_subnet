import numpy as np

from chunking.validator.reward import rank_responses

def test_rank_responses():
    
    # ranks should be in descending order
    rewards = np.arange(20) + 1
    process_times = np.arange(20) + 0.2
    
    ranked_responses = rank_responses(rewards, process_times)
    
    expected_ranks = np.arange(20)[::-1]
    
    print(ranked_responses)
    print(expected_ranks)
    
    assert np.array_equal(ranked_responses, expected_ranks)
    
    # all ranks should be -1
    
    rewards = np.repeat(0, 20)
    process_times = np.repeat(np.inf, 20)
    
    ranked_responses = rank_responses(rewards, process_times)        
    
    expected_ranks = np.repeat(-1, 20)
    
    assert np.array_equal(ranked_responses, expected_ranks)
    
    # test tiebreaking by process time
    
    rewards = np.repeat(1.49259282339, 20)
    process_times = np.arange(20) + 0.2
    
    ranked_responses = rank_responses(rewards, process_times)
    
    expected_ranks = np.arange(20)
    
    assert np.array_equal(ranked_responses, expected_ranks)
    
    # should cut off at 6th decimal place and rank in ascending order
    
    rewards = np.concatenate([np.repeat(1.49259282338, 10), np.repeat(1.49259282339, 10)])
    process_times = np.arange(20)
    
    ranked_responses = rank_responses(rewards, process_times)
    
    expected_ranks = np.arange(20)

    assert np.array_equal(ranked_responses, expected_ranks)
    
    # test multiple tie breaks
    
    rewards = np.concatenate([np.repeat(1.3, 10), np.repeat(1.2, 10)])
    process_times = np.arange(20)
    
    ranked_responses = rank_responses(rewards, process_times)
    
    expected_ranks = np.arange(20)
    
    assert np.array_equal(ranked_responses, expected_ranks)

    
    rewards = np.repeat(1.0, 5)
    process_times = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    ranked_responses = rank_responses(rewards, process_times)
    expected_ranks = np.array([4, 3, 2, 1, 0])
    assert np.array_equal(ranked_responses, expected_ranks), "Equal rewards, varying process times test failed"

    # rewards with many decimal places, should all tiebreak
    rewards = np.array([1.123456786, 1.123456787, 1.123456788, 1.123456789])
    process_times = np.array([1.0, 2.0, 3.0, 4.0])
    ranked_responses = rank_responses(rewards, process_times)
    expected_ranks = np.array([0, 1, 2, 3])
    assert np.array_equal(ranked_responses, expected_ranks), "Many decimal places test failed"        

    # edge case with very close rewards and process times
    rewards = np.array([1.000001, 1.000002, 1.000003])
    process_times = np.array([0.000001, 0.000002, 0.000003])
    ranked_responses = rank_responses(rewards, process_times)
    expected_ranks = np.array([2, 1, 0])
    assert np.array_equal(ranked_responses, expected_ranks)    
    
    # same rewards, same process times, should break on uid desc
    rewards = np.array([1.0, 1.0, 1.0])
    process_times = np.array([0.0, 0.0, 0.0])
    ranked_responses = rank_responses(rewards, process_times)
    expected_ranks = np.array([2, 1, 0])
    assert np.array_equal(ranked_responses, expected_ranks), "Same rewards, same process times test failed"