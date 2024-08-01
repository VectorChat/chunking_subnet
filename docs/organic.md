# Organic Queries

The primary goal of this subnet is to serve organic queries, as the leading provider of the best intelligent chunking.

## Task API

The Task API is how the [default validator](../docs/validator.md) handles organic and synthetic queries. If `--accept_organic_queries` is `true` it will first check, every time step (default: seconds), if it has received an organic query from its API host. If it hasn't, it will generate a [synthetic query](./synthetic.md).

From [task_api.py](../chunking/validator/task_api.py), the validator first checks if it has received an organic request:
```python
def get_new_task(self, validator: Validator):

        if os.environ.get('ALLOW_ORGANIC_CHUNKING_QUERIES') == 'True':
            hotkey = validator.wallet.get_hotkey()
            nonce = validator.step
            data = {
                'hotkey_address': hotkey.ss58_address,
                'nonce': nonce
            }

            # sign request with validator hotkey
            request_signature = sign(
                (hotkey.public_key, hotkey.private_key),
                str.encode(json.dumps(data))
                ).hex()

            API_host = os.environ['CHUNKING_API_HOST']
            task_url = f"{API_host}/task_api/get_new_task/"
            headers = {"Content-Type": "application/json"}
            request_data = {
                'data': data, 
                'signature': request_signature
                }
```

## Chunking.com

The default value for `--api_host` is currently set to none, but will change to the Chunking.com Task API in the near future. This serves as an easy to opt-in network for validators to sell their bandwidth and receive compensation in return.

## Custom Task API

A framework to easily create your own network will be released around the same time as the Chunking.com Task API. Validators can then serve their own organic queries, or market their bandwidth independently.