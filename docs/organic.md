# Organic Queries

The primary goal of this subnet is to serve organic queries, as the leading provider of the best intelligent chunking.

## Task API

When the `--enable_task_api` flag is set, the validator will be able to serve organic queries from external clients. This works by having the validator process run a "sidecar" api server
that can accept and serve requests as they come in (the impl can be found [here](../chunking/validator/integrated_api.py)). The host and port can be configured with `--task_api.host` and `--task_api.port` respectively.

After running, the swagger UI for the task API can be viewed at `http://<HOST>:<PORT>/docs`.

There are three endpoints at the moment (further described in the swagger UI):

| Method | Endpoint    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`  | `/rankings` | Returns the rankings of miners in the validator's tournament via two mappings (ranking -> UID and UID -> ranking). These mappings are in array form.                                                                                                                                                                                                                                                                                                                                                                                            |
| `GET`  | `/groups`   | Returns the groups of miners in the validator's tournament via two mappings (group index -> group UIDs and UID -> group indices). These mappings are in array form.                                                                                                                                                                                                                                                                                                                                                                             |
| `POST` | `/chunk`    | Accepts a chunk request and processes it. The user can specify the document, chunk size, chunk quantity, and time parameters (timeout and time soft max). The user can also specify either a single miner to query or a specific miner group to query. If no miner or group is specified, the request will be processed by a random miner group. The user can choose whether or not the request should be scored and count towards the tournament ranking (if only one miner is queried than this cannot count towards the tournament ranking). |

> [!NOTE]
> There is currently no built in way to blacklist/whitelist entities from using the task API. It is suggested to use a firewall like `ufw` to block requests from unwanted entities and/or only allow requests from predefined IPs.
