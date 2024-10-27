import time
from typing import List, Optional
from fastapi import Body, HTTPException
from pydantic import BaseModel, Field
import bittensor as bt
import traceback
from chunking.utils.integrated_api.chunk import (
    ChunkRequest,
    ChunkResponse,
    chunk_handler,
)
from chunking.utils.integrated_api.log import api_log
from chunking.validator.tournament import get_miner_groups


class RankingsResponse(BaseModel):
    rankings: List[int] = Field(
        ...,
        description="List of miner rankings by global tournament rank. Index is miner's global ranking, value is uid.",
    )
    by_uid: List[int] = Field(
        ...,
        description="List of miner uids by global tournament rank. Index is miner UID, value is miner's global ranking.",
    )


class GroupsResponse(BaseModel):
    groups: List[List[int]] = Field(
        ...,
        description="List of miner groups. Index is group index, value is miner UIDs in the group.",
    )
    by_uid: List[List[int]] = Field(
        ...,
        description="List of miner UIDs by group. Index is miner UID, value is group indices that the miner belongs to.",
    )


def setup_routes(self):

    @self.app.get("/rankings")
    async def rankings() -> RankingsResponse:
        rankings = self.rankings.tolist()
        by_uid_mapping = {uid: rank for rank, uid in enumerate(self.rankings)}
        by_uid = [by_uid_mapping[uid] for uid in range(len(self.rankings))]
        api_log(f"Rankings: {rankings}")
        api_log(f"By UID: {by_uid}")
        return RankingsResponse(rankings=rankings, by_uid=by_uid)

    @self.app.get("/groups")
    async def groups() -> GroupsResponse:
        groups, _, _ = get_miner_groups(self)
        by_uid_mapping = {}
        for group_index, miner_group_uids in enumerate(groups):
            for uid in miner_group_uids:
                uid_int = uid.item()
                if uid_int not in by_uid_mapping:
                    by_uid_mapping[uid_int] = []
                by_uid_mapping[uid_int].append(group_index)
        by_uid = [by_uid_mapping[uid] for uid in range(len(self.rankings))]
        api_log(f"Groups: {groups}")
        api_log(f"By UID: {by_uid}")
        return GroupsResponse(groups=groups, by_uid=by_uid)

    @self.app.post("/chunk")
    async def chunk(request: ChunkRequest) -> ChunkResponse:
        try:
            start_time = time.time()
            result = await chunk_handler(self, request)
            end_time = time.time()
            bt.logging.debug(
                f"Serviced request in {end_time - start_time} seconds for doc of length {len(request.document)}"
            )
            return result
        except Exception as e:
            bt.logging.error(f"Error in chunking: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Internal server error")

    bt.logging.info("Chunking API setup complete")
