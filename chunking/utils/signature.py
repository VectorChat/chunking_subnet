from sr25519 import verify
from scalecodec.utils.ss58 import ss58_decode

def verify_signature(signature: str, data: str, hotkey_ss58: str):
    public_key = ss58_decode(hotkey_ss58)
    verified = verify(bytes.fromhex(signature), str.encode(data), bytes.fromhex(public_key))
    return verified

