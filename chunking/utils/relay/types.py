from datetime import datetime
from pydantic import BaseModel

from chunking.utils.ipfs.types import IPFSPin


class RelayMessage(BaseModel):
    document_hash: str
    embeddings: list[list[float]]


class RelayPayload(BaseModel):
    message: RelayMessage
    signature: str


class IPFSRelayPin(IPFSPin):
    payload: RelayPayload
