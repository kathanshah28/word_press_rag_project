import html
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.post import CleanedPost

# Load environment variables from .env
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def clean_text(html_content: str) -> str:
    """
    Strips WordPress Gutenberg comments, HTML tags, decodes HTML entities,
    and normalizes whitespace into clean plain text.
    """
    if not html_content or not isinstance(html_content, str):
        return ""

    # Remove WordPress Gutenberg comments e.g. <!-- wp:paragraph -->
    cleaned = re.sub(r"<!--\s*/?wp:.*?-->", "", html_content, flags=re.DOTALL)

    # Extract plain text from HTML
    soup = BeautifulSoup(cleaned, "html.parser")
    text = soup.get_text(separator="\n")

    # Unescape HTML entities (e.g. &amp; -> &, &#8217; -> ')
    text = html.unescape(text)

    # Normalize excessive newlines and leading/trailing whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def fetch_posts(
    post_type: str = "posts",
    per_page: int = 100,
    max_pages: Optional[int] = None,
) -> List[CleanedPost]:
    """
    Fetches data from the WordPress REST API specified in .env, extracts
    only clean text (post_id, title, link, content), and validates them into
    Pydantic CleanedPost objects ready for embedding generation.

    Args:
        post_type: The WordPress endpoint to fetch (e.g., 'posts', 'pages').
        per_page: Number of items per request page (WordPress max is 100).
        max_pages: Optional maximum number of pages to fetch (None for all).

    Returns:
        List[CleanedPost]: List of validated Pydantic CleanedPost objects.
    """
    base_url = os.getenv("WORDPRESS_URL", "").rstrip("/")
    if not base_url:
        raise ValueError("WORDPRESS_URL is not set in the .env file.")

    username = os.getenv("WORDPRESS_USERNAME")
    password = os.getenv("WORDPRESS_APP_PASSWORD")
    auth = (username, password) if (username and password) else None

    endpoint = f"{base_url}/wp-json/wp/v2/{post_type}"
    cleaned_posts: List[CleanedPost] = []
    page = 1

    print(f"Fetching '{post_type}' from {endpoint} ...")

    while True:
        params = {
            "page": page,
            "per_page": per_page,
            "status": "publish",
        }

        try:
            response = requests.get(endpoint, params=params, auth=auth, timeout=15)

            # WordPress REST API returns 400 when page index exceeds available pages
            if response.status_code == 400:
                break

            response.raise_for_status()
            items = response.json()

            if not items or not isinstance(items, list):
                break

            for item in items:
                raw_title = item.get("title", {}).get("rendered", "") if isinstance(item.get("title"), dict) else str(item.get("title") or "")
                raw_content = item.get("content", {}).get("rendered", "") if isinstance(item.get("content"), dict) else str(item.get("content") or "")

                # Instantiate Pydantic model for clean data management and validation
                post_obj = CleanedPost(
                    post_id=item.get("id"),
                    title=clean_text(raw_title),
                    link=item.get("link", ""),
                    content=clean_text(raw_content),
                )
                cleaned_posts.append(post_obj)

            print(f"  -> Page {page}: processed {len(items)} items")

            # Check total pages from WordPress response headers
            total_pages = int(response.headers.get("X-WP-TotalPages", 1))
            if page >= total_pages or (max_pages and page >= max_pages):
                break

            page += 1

        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break

    print(f"Total {len(cleaned_posts)} {post_type} cleaned and validated successfully.\n")
    return cleaned_posts


### testing
if __name__ == "__main__":
    posts = fetch_posts(post_type="posts")

    for i, post in enumerate(posts, 1):
        print(f"--- [Post {i}] ---")
        print(f"Post ID : {post.post_id}")
        print(f"Title   : {post.title}")
        print(f"Link    : {post.link}")
        print(f"Content :\n{post.content[:]}")  # preview first 200 chars
        # print(f"LangChain Doc:\n{post.to_document()}")
        print("-" * 40)
