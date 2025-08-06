import requests
from bs4 import BeautifulSoup

def scrape_reddit_titles(subreddit, limit=5):
    url = f"https://www.reddit.com/r/{subreddit}/hot/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Failed to fetch Reddit page:", response.status_code)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    titles = []

    for post in soup.find_all("h3"):
        titles.append(post.text)
        if len(titles) >= limit:
            break

    return titles

# Test
print("✅ Top posts from r/stocks:\n")
for t in scrape_reddit_titles("stocks"):
    print("-", t)
