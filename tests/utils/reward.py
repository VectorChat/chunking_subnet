
from openai import AsyncOpenAI

from chunking.protocol import chunkSynapse
from chunking.validator.reward import reward


async def calc_reward(document: str, chunk_size: int, chunk_qty: int, client: AsyncOpenAI, chunks: list[str], num_embeddings: int = 2000, verbose: bool = False):

    res_synapse = chunkSynapse(
        document=document,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        time_soft_max=15,
        timeout=20,
        chunks=chunks,
    )

    reward_value, extra_info = await reward(
        document=document,
        chunk_size=chunk_size,
        chunk_qty=chunk_qty,
        response=res_synapse,
        num_embeddings=num_embeddings,
        client=client,
        verbose=verbose,
    )

    return reward_value, extra_info