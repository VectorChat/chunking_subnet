import bittensor as bt


def api_log(message: str):
    bt.logging.info(f"[TASK_API]: {message}")
