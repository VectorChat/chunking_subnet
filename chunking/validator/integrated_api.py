
from typing import List, Optional

from neurons.validator import Validator


class ChunkRequest(BaseModel):
    document: str
    chunk_size: int
    chunk_qty: int
    miner_uids: Optional[List[int]] = None
    do_grading: bool = False

def setup_routes(self: Validator):
    @self.app.post("/chunk")
    async def chunk(request: ChunkRequest):
