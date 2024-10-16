from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class IPFSPin(BaseModel):
    """
    An IPFS pin object, holding necessary attributes for a pin.
    """

    cid: str
    created_at: datetime
    raw_content: Optional[str] = None
