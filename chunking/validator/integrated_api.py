from typing import List, Optional
from fastapi import Body, HTTPException
from pydantic import BaseModel, Field
import bittensor as bt
import traceback
from chunking.protocol import chunkSynapse
from chunking.utils.relay.relay import make_relay_payload
from chunking.validator.tournament import get_miner_groups, run_tournament_round


class ChunkRequest(BaseModel):
    document: str
    chunk_size: int
    chunk_qty: Optional[int] = Body(
        default=None,
        description="Max number of chunks to create, `ceil(ceil(len(document) / chunk_size) * 1.5)",
    )
    timeout: Optional[float] = Body(
        default=60, description="Timeout for the chunking task"
    )
    miner_index: Optional[int] = Body(
        default=None,
        description="Specific miner index to query (index is miner's global ranking in tournament)",
    )
    miner_group_index: Optional[int] = Body(
        default=None, description="Specific miner group index to query"
    )
    do_scoring: bool = Body(
        default=True,
        description="Whether chunks should count towards scores in the tournament",
    )


class ChunkResult(BaseModel):
    chunks: List[str]
    miner_signature: str
    uid: int
    miner_group_index: int


class ChunkResponse(BaseModel):
    results: List[ChunkResult]


class RankingsResponse(BaseModel):
    rankings: List[int] = Field(...,
        description="List of miner rankings by global tournament rank. Index is miner's global ranking, value is uid."
    )
    by_uid: List[int] = Field(...,
        description="List of miner uids by global tournament rank. Index is miner UID, value is miner's global ranking."
    )

class GroupsResponse(BaseModel):
    groups: List[List[int]] = Field(..., description="List of miner groups. Index is group index, value is miner UIDs in the group.")
    by_uid: List[List[int]] = Field(..., description="List of miner UIDs by group. Index is miner UID, value is group indices that the miner belongs to.")

def setup_routes(self):

    @self.app.get("/rankings")
    async def rankings() -> RankingsResponse:
        rankings = self.rankings.tolist()
        by_uid_mapping = {uid: rank for rank, uid in enumerate(self.rankings)}
        by_uid = [by_uid_mapping[uid] for uid in range(len(self.rankings))]
        return RankingsResponse(rankings=rankings, by_uid=by_uid)

    @self.app.get("/groups")
    async def groups() -> GroupsResponse:
        groups, _, _ = get_miner_groups(self)
        by_uid_mapping = {}
        for group_index, miner_group_uids in enumerate(groups):
            for uid in miner_group_uids:
                if uid not in by_uid_mapping:
                    by_uid_mapping[uid] = []
                by_uid_mapping[uid].append(group_index)
        return GroupsResponse(groups=groups, by_uid=by_uid_mapping)

    @self.app.post("/chunk")
    async def chunk(request: ChunkRequest) -> ChunkResponse:
        try:
            input_synapse = chunkSynapse(
                document=request.document,
                chunk_size=request.chunk_size,
                chunk_qty=request.chunk_qty,
                timeout=request.timeout,
                time_soft_max=request.timeout * 0.75,
            )

            CID = await make_relay_payload(
                input_synapse.document, self.aclient, self.wallet
            )

            input_synapse.CID = CID

            results = await run_tournament_round(
                self, input_synapse, request.miner_uid, request.miner_group_index
            )

            usable_results = [result for result in results if result is not None]

            # TODO: make background task if possible
            #
            chunk_results: List[ChunkResult] = []
            for result in usable_results:
                if request.do_scoring:
                    await self.queue_score_update(result)

                miner_group_uids = result.miner_group_uids

                for response, miner_group_uid in zip(
                    result.responses, miner_group_uids
                ):
                    if response.chunks and response.miner_signature:
                        chunk_results.append(
                            ChunkResult(
                                chunks=response.chunks,
                                miner_signature=response.miner_signature,
                                uid=miner_group_uid,
                                miner_group_index=result.miner_group_index,
                            )
                        )

            return ChunkResponse(results=chunk_results)
        except Exception as e:
            bt.logging.error(f"Error in chunking: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Internal server error")

    bt.logging.info("Chunking API setup complete")
