import traceback
import requests


def get_articles():
    try:
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
        return articles
    except Exception as e:
        print(f"Error syncing articles: {e}")
        traceback.print_exc()
        return []
