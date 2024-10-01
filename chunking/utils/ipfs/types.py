from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class IPFSPin(BaseModel):
    cid: str
    created_at: datetime
    raw_content: Optional[str] = None
