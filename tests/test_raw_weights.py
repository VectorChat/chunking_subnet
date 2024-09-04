import numpy as np
from chunking.base.validator import BaseValidatorNeuron

def test_raw_weights():
    
    scores = np.array([1, 2, 3, 4, 5]) / 2
    rankings = np.argsort(scores)
    
    raw_weights = BaseValidatorNeuron._get_raw_weights(scores, rankings)
    
    # should equal 1/2 ^ i
    assert np.allclose(raw_weights, 1 / 2 ** rankings)
        
    scores = np.arange(1, 100)    
    rankings = np.argsort(scores)
    
    raw_weights = BaseValidatorNeuron._get_raw_weights(scores, rankings)

    # first seven should be 1/2 ^ i
    assert np.allclose(raw_weights[:7], 1 / 2 ** np.arange(7))
    
    # eighth should not be 1/2 ^ i
    assert raw_weights[7] != 1 / 2 ** 7
    
    rest_sum = np.sum(raw_weights[7:])
    
    print(f"rest_sum (from i = 7 to end) = {rest_sum}")
    print(f"rest sum from i = 8 to end = {np.sum(raw_weights[8:])}")    
    
    print(f"1/2 ^ 7 = {1 / 2 ** 7}")
    
    # should be equal up to fourth decimal place
    assert np.allclose(rest_sum, 1 / 2 ** 7, atol=1e-4)
    
    
    # test with all infinity scores
    scores = np.inf * np.ones(100)
    # rankings should be all -1
    rankings = np.repeat(-1, 100)
    
    raw_weights = BaseValidatorNeuron._get_raw_weights(scores, rankings)
    
    assert np.allclose(raw_weights, np.zeros(100))
    
    #
    scores = np.inf * np.ones(100)
    
    # test with only 5 people having scores
    
    top_5 = np.arange(5)
    
    scores[top_5] = np.arange(1, 6)
    
    print(f"scores: {scores}")
    
    rankings = np.argsort(scores)
    
    raw_weights = BaseValidatorNeuron._get_raw_weights(scores, rankings)
    
    print(f"raw_weights: {raw_weights}")
    print(f"sum of raw_weights: {np.sum(raw_weights)}")
    
    assert np.allclose(raw_weights[:5], 1 / 2 ** top_5)
    assert np.sum(raw_weights[5:]) == 0
    
    
    # test with only 1 person having a score
    
    scores = np.inf * np.ones(100)
    scores[0] = 1
    
    rankings = np.argsort(scores)
    
    raw_weights = BaseValidatorNeuron._get_raw_weights(scores, rankings)
    
    print(f"raw_weights: {raw_weights}")
    print(f"sum of raw_weights: {np.sum(raw_weights)}")
    
    assert raw_weights[0] == 1
    assert np.sum(raw_weights[1:]) == 0
    
    # test with 201 active miners
    np.random.seed(42)
    scores = np.random.rand(201)
    rankings = np.argsort(scores)
    
    raw_weights = BaseValidatorNeuron._get_raw_weights(scores, rankings)
    
    print(f"raw_weights: {raw_weights}")
    print(f"sum of raw_weights: {np.sum(raw_weights)}")        
    
    top_7_uids = rankings[:7]
        
    # top 7 should be 1/2 ^ i
    assert np.allclose(raw_weights[top_7_uids], 1 / 2 ** np.arange(7))
        
    rest_uids = rankings[7:]
    
    rest_sum = np.sum(raw_weights[rest_uids])
    
    print(f"rest_sum (from rank = 7 to end) = {rest_sum}")
    
    print(f"1/2 ^ 7 = {1 / 2 ** 7}")
    
    # last should sum to 1/2 ^ 7 with some error
    assert np.allclose(rest_sum, 1 / 2 ** 7, atol=1e-4)