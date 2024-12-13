import argparse
import asyncio
import random
import time
from chunking.utils.integrated_api.chunk.types import ChunkResponse
from tests.test_integrated_api import get_doc_and_query
from tests.utils.articles import get_articles


async def main(args: argparse.Namespace | None = None):
    print("Getting test pageids")
    test_pageids = get_articles()

    print(f"Got {len(test_pageids)} test pageids")

    if args is None:
        num_articles = 5
        batch_size = 1
        group_indices = None
        custom_uids = [16, 17, 18]
        timeout = 20
        sleep_time = 0.0
    else:
        num_articles = args.num_articles
        batch_size = args.batch_size
        group_indices = args.group_indices
        custom_uids = args.custom_uids
        timeout = args.timeout
        sleep_time = args.sleep_time

    if not group_indices and not custom_uids:
        print("Must provide either group indices or custom uids")
        exit(1)

    if group_indices and custom_uids:
        print("Cannot provide both group indices and custom uids")
        exit(1)

    for i in range(0, num_articles, batch_size):
        batch_pageids = test_pageids[i : i + batch_size]

        coros = [
            get_doc_and_query(
                pageid,
                random.choice(group_indices) if group_indices is not None else None,
                custom_miner_uids=custom_uids,
                timeout=timeout,
            )
            for pageid in batch_pageids
        ]

        responses = await asyncio.gather(*coros)

        for j, response in enumerate(responses):
            chunk_response = ChunkResponse.model_validate(response.json())

            is_error = False

            uids_seen = set()

            for result in chunk_response.results:
                uids_seen.add(result.uid)

                if result.chunks is None:
                    print(
                        f"Got None chunks for pageid {batch_pageids[j]}, uid: {result.uid} in group {result.miner_group_index}. Process time: {result.process_time}"
                    )
                    is_error = True
                else:
                    print(
                        f"Got {len(result.chunks)} chunks for pageid {batch_pageids[j]}, uid: {result.uid} in group {result.miner_group_index}. Process time: {result.process_time}"
                    )

            if custom_uids is not None:
                if len(uids_seen) != len(custom_uids):
                    print(f"Missing uids: {set(custom_uids) - uids_seen}")
                    raise ValueError(f"Missing uids: {set(custom_uids) - uids_seen}")

            if is_error:
                raise ValueError("Error(s) found in responses")

        time.sleep(sleep_time)


def test_queries():
    asyncio.run(main())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_articles", "-n", type=int, default=5)
    argparser.add_argument("--batch_size", "-b", type=int, default=1)
    argparser.add_argument("--sleep_time", "-s", type=float, default=0.0)
    argparser.add_argument("--group_indices", "-g", type=int, nargs="+", default=None)
    argparser.add_argument("--custom_uids", "-c", type=int, nargs="+", default=None)
    argparser.add_argument("--timeout", "-t", type=float, default=20)
    argparser.add_argument(
        "--time_soft_max_multiplier", "-tm", type=float, default=0.75
    )

    args = argparser.parse_args()

    main(args)
