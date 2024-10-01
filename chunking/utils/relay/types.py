from datetime import datetime
from pydantic import BaseModel

from chunking.utils.ipfs.types import IPFSPin


class RelayMessage(BaseModel):
    """
    A relay message is a message that is sent to the miner to check for exact and 'fuzzy' duplicates.
    It features a hash of the document, and a list of embeddings for the document.
    """

    document_hash: str
    embeddings: list[list[float]]


class RelayPayload(BaseModel):
    """
    The payload has the message and a signature from the validator to ensure authenticity.
    """

    message: RelayMessage
    signature: str


class IPFSRelayPin(IPFSPin):
    """
    An IPFS pin object that also includes the relay payload.
    """

    payload: RelayPayload
