from random import sample
from openai import OpenAI
from chunking.validator.reward import reward
from chunking.validator.task_api import generate_synthetic_synapse
from nltk.tokenize import sent_tokenize

def base_chunker(text: str, chunk_size: int):
    document = sent_tokenize(text)

    chunks = []       
    while len(document) > 0:
        chunks.append(document[0])
        del document[0]
        while len(document) > 0:
            if len(chunks[-1] + " " + document[0]) > chunk_size:
                break
            chunks[-1] += (" " + document.pop(0))

    return chunks

def test_reward_fn():

    tuple = generate_synthetic_synapse(None, 33653136, 20)

    synapse = tuple[0]    

    assert synapse.time_soft_max == 15

    client = OpenAI()

    NUM_EMBEDDINGS = 150

    # reward should be 0 if no chunks

    reward_value, _ = reward(None, synapse.document, synapse.chunk_size, synapse.chunk_qty, synapse, client, NUM_EMBEDDINGS)

    assert reward_value == 0


    # reward should be zero if any word is reordered

    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    ## shuffle the words in the first chunk

    test_chunks[0] = ' '.join(sample(test_chunks[0].split(), len(test_chunks[0].split())))

    synapse.chunks = test_chunks

    reward_value, _ = reward(None, synapse.document, synapse.chunk_size, synapse.chunk_qty, synapse, client, NUM_EMBEDDINGS)

    assert reward_value == 0

    # reward should be zero if any word is removed

    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    ## remove the first word from the first chunk

    test_chunks[0] = ' '.join(test_chunks[0].split()[1:])

    synapse.chunks = test_chunks

    reward_value, _ = reward(None, synapse.document, synapse.chunk_size, synapse.chunk_qty, synapse, client, NUM_EMBEDDINGS)

    assert reward_value == 0

    # reward should be zero if any word is added

    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    ## add a word to the first chunk

    test_chunks[0] = ' '.join(test_chunks[0].split() + ['word'])

    synapse.chunks = test_chunks

    reward_value, _ = reward(None, synapse.document, synapse.chunk_size, synapse.chunk_qty, synapse, client, NUM_EMBEDDINGS)

    assert reward_value == 0

    # reward should be zero if any chunks are removed

    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    ## remove the first chunk

    test_chunks = test_chunks[1:]

    synapse.chunks = test_chunks

    reward_value, _ = reward(None, synapse.document, synapse.chunk_size, synapse.chunk_qty, synapse, client, NUM_EMBEDDINGS)

    assert reward_value == 0

    # should give reward for proper chunking

    test_chunks = base_chunker(synapse.document, synapse.chunk_size)

    synapse.chunks = test_chunks

    reward_value, _ = reward(None, synapse.document, synapse.chunk_size, synapse.chunk_qty, synapse, client, NUM_EMBEDDINGS)

    assert reward_value > 0