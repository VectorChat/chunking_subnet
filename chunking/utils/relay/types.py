from pydantic import BaseModel


class RelayMessage(BaseModel):
    document_hash: str
    embeddings: list[list[float]]

class RelayPayload(BaseModel):
    message: RelayMessage
    signature: str

