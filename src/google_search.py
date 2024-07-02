import argparse
import json
import os
from datetime import datetime

import dateparser
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from trafilatura import fetch_url, extract


def google_search(args):
    url = f"https://google.serper.dev/{args.type}"

    payload = json.dumps({
        "q": args.query,
        "location": args.location,
        "gl": args.country,
        "hl": args.language,
        "num": args.nums,
        "tbs": args.tbs,
        "page": args.page,
        "autocorrect": args.autocorrect
    })
    headers = {
        'X-API-KEY': os.environ['X_API_KEY'],
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    result = json.loads(response.text)
    # update the date field of news
    if args.type == "news":
        for news in result["news"]:
            news["date"] = dateparser.parse(news["date"]).strftime("%Y-%m-%dT%H:%M:%S")

    # dig the link of news
    if args.type == "news":
        for news in tqdm(result["news"], desc="Mining content for news"):
            downloaded = fetch_url(news["link"])
            content = extract(downloaded)
            news["content"] = content

    now = datetime.now()
    result.update({"created_at": now.strftime("%Y-%m-%dT%H:%M:%S")})
    output_filename = \
        (f"{args.query.replace(' ', '_')}-{args.country}-{args.language}-{str(args.autocorrect).lower()}-"
         f"{args.page}-{args.nums}-{args.type}-{args.location}.json") if args.output is None else args.outout
    with open(output_filename, "w") as f:
        json.dump(result, f)


def cli_args():
    parser = argparse.ArgumentParser(description="Google Search")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--type", default="search", help="Type of search, choices: `search`, `news` and etc")
    parser.add_argument("--location", default="United States", help="Search location")
    parser.add_argument("--country", default="us", help="Country")
    parser.add_argument("--language", default="en", help="Language")
    parser.add_argument("--autocorrect", default=True, help="Auto Correct")
    parser.add_argument("--nums", default=100, help="Number of results")
    parser.add_argument("--page", default=1, help="Page number")
    parser.add_argument("--tbs", default="qdr:m", help="Time period")
    parser.add_argument("--output", default=None, help="Output file")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    google_search(cli_args())
