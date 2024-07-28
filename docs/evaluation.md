# Reward Function

This subnet incentivizes responses that maximize intrachunk similarity and interchunk dissimilarity, without exceeding variable constraints, such as time or maximum chunk length. This subnet measures similarity by the dot product.

Since the similarity score can be influenced significantly by the content in the provided document, evaluation is always done relatively, within groups of miners that all faced the same query. See [Incentive Mechanism](./incentive_mechanism.md) to learn more.

In the default validator, [reward.py](../chunking/validator/reward.py) scores the responses of each miner.

## Failure States

### 1. No new words
First, the validator confirms that each word in the chunked response also exists, in the same order, in the original document.
```python
    # check that every word in chunk exists and is in the same order as the source document
    chunk_words = ' '.join(word_tokenize(chunks[i]))
    combined_chunk_words += ' ' + chunk_words
    if chunk_words not in ' '.join(document_words):
        return 0
```
### 2. All words present
Then, the validator confirms that every set of 3 adjacent words in the original document is also present within the chunked response.
```python
    # check that every set of 3 adjacent words from the document appears in the chunks
    for i in range(0, len(document_words), 3):
        if (len(' '.join(document_words[i:i+3])) < chunk_size
            and ' '.join(document_words[i:i+3]) not in combined_chunk_words):
            return 0
```

If either don't hold true, the score is 0.

## Evaluating

After passing the fail states, the validator parses through each chunk, creating 'small chunks' of 3 sentences or fewer.

```python
    # create test segments
    sentences = sent_tokenize(chunks[i])
    for j in range(0, len(sentences), 3):
        text = " ".join(sentences[j:j+3])
        smallChunks.append(smallChunk(i, text))
```

A random sample, of `num_embeddings` size, is taken and then embedded. The default value is 50.

```python
    # pick out segments to use for evaluation
    if num_embeddings < len(smallChunks):
        testChunks = sample(smallChunks, num_embeddings)
    else:
        testChunks = smallChunks
```

Then, to calculate the similarity score, the dot product of every possible pair of embeddings is calculated. For each pair that originated from the same chunk, the result is added to the score (intrachunk similarity). For each pair originating from different chunks, the result is subtracted from the score (interchunk dissimilarity).

```python
    for i in range(len(testChunks) - 1):
        j = i + 1
        while j < len(testChunks):
            if testChunks[i].sourceChunk == testChunks[j].sourceChunk:
                reward += np.dot(np.asarray(embeddings[i]), np.asarray(embeddings[j]))
            else:
                reward -= np.dot(np.asarray(embeddings[i]), np.asarray(embeddings[j]))
            j += 1  
```

Here is a visualization of how the validator calculates a miner’s score:

![evaluations](../assets/evaluations.png)

# Penalties

Finally, penalities are deducted from the score.

Responses are penalized exponentially for each character over the maximum chunk length.
```python
    # add up size penalty to be applied later
    chunk_length = len(chunks[i])
    if chunk_length > chunk_size:            
        size_penalty += ((chunk_length / chunk_size) - 1) * 10            
        _verbose(f"Chunk {i} is too long: {chunk_length} characters, new size penalty: {size_penalty}")     

```
```python
    reward *= (2/3) ** size_penalty  
```


Finally, note that there is a soft-time limit (default: 5 seconds). Validators exponentially penalize responses for each second they are late.
```python
    if response.dendrite.process_time > response.time_soft_max:
        over_time = response.dendrite.process_time - response.time_soft_max
        _verbose(f"Applying time penalty: {over_time} seconds over time")
        reward *= (2/3) ** over_time
```