import json
import os
import traceback
import requests

cache_file = "tests/assets/articles.json"


def get_article_ids_from_cache():
    try:
        with open(cache_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_article_ids_to_cache(article_ids: list[int]):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(article_ids, f)


def get_articles():
    try:
        article_ids = get_article_ids_from_cache()
        if article_ids:
            return article_ids
        articles = []
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmpageid": "8966941",
                "cmprop": "ids",
                "cmlimit": "max",
            },
        ).json()

        articles.extend(
            [page["pageid"] for page in response["query"]["categorymembers"]]
        )
        continuation = response.get("continue")
        while continuation is not None:
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "list": "categorymembers",
                    "cmpageid": "8966941",
                    "cmprop": "ids",
                    "cmlimit": "max",
                    "cmcontinue": continuation.get("cmcontinue"),
                },
            ).json()
            continuation = response.get("continue")
            articles.extend(
                [page["pageid"] for page in response["query"]["categorymembers"]]
            )
        save_article_ids_to_cache(articles)
        return articles
    except Exception as e:
        print(f"Error syncing articles: {e}")
        traceback.print_exc()
        return []
