import argparse
import json
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://millercenter.org"
PRESIDENTS_LIST_PATH = Path(__file__).parent / "miller_center_presidents.json"
TEXT_DIR = Path(__file__).parent.parent / "text"
REQUEST_DELAY_SECONDS = 1.0

SUBPAGES = [
    "life-in-brief",
    "life-before-the-presidency",
    "campaigns-and-elections",
    "domestic-affairs",
    "foreign-affairs",
    "life-after-the-presidency",
    "family-life",
    "the-american-franchise",
    "impact-and-legacy",
]


def fetch_html(url: str) -> BeautifulSoup:
    """Fetch the HTML content from the given URL and return as a BeautifulSoup object."""
    response = httpx.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


_HEADING_PREFIX = {
    "h1": "#",
    "h2": "##",
    "h3": "###",
    "h4": "####",
    "h5": "#####",
    "h6": "######",
}


def parse_miller_center_article(soup: BeautifulSoup) -> dict:
    """Parse a Miller Center article page BeautifulSoup object into structured fields."""
    article_el = soup.find(class_="simple-article")
    if not article_el:
        return {
            "title": "",
            "author": "",
            "text": "",
        }

    title_el = article_el.select_one(".article-title")
    title = title_el.get_text(strip=True) if title_el else ""

    author_el = article_el.select_one(".article-author")
    author = ""
    if author_el:
        author_span = author_el.find("span")
        if author_span:
            author = author_span.get_text(strip=True)
        else:
            author = (
                author_el.get_text(separator=" ", strip=True)
                .removeprefix("By ")
                .strip()
            )

    body_el = article_el.select_one(
        ".main-body-wrapper .paragraph--type--wysiwyg-body"
    )

    blocks: list[str] = []
    if body_el:
        for child in body_el.children:
            if not isinstance(child, Tag):
                continue
            text = child.get_text(strip=True)
            if not text:
                continue
            prefix = _HEADING_PREFIX.get(child.name.lower())
            blocks.append(f"{prefix} {text}" if prefix else text)

    # Prepend title
    if title:
        blocks = [f"# {title}"] + blocks

    return {
        "title": title,
        "author": author,
        "text": "\n\n".join(blocks),
    }


def normalize_name(name: str) -> str:
    return name.replace(" ", "_").replace(".", "").lower()


def build_filename(
    number: int, name: str, subpage_index: int, subpage: str
) -> str:
    return f"{number:02d}_{normalize_name(name)}_{subpage_index}_{subpage}.txt"


def scrape_article(president_link: str, subpage: str) -> dict:
    url = f"{BASE_URL}{president_link}/{subpage}"
    soup = fetch_html(url)
    return parse_miller_center_article(soup)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Miller Center president articles into text files."
    )
    parser.add_argument(
        "-s",
        "--subfolder",
        help="Name of the subfolder inside text/ where results should be saved.",
    )
    args = parser.parse_args()

    output_dir = TEXT_DIR / args.subfolder
    if output_dir.exists():
        raise FileExistsError(
            f"Folder '{output_dir}' already exists. Aborting to prevent overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    with open(PRESIDENTS_LIST_PATH) as f:
        presidents = json.load(f)

    total = len(presidents) * len(SUBPAGES)
    count = 0

    for president in presidents:
        for subpage_index, subpage in enumerate(SUBPAGES):
            count += 1
            filename = build_filename(
                president["number"], president["name"], subpage_index, subpage
            )
            print(
                f"[{count:4}/{total}] Scraping... "
                f"{president['name']:25} / {subpage}"
            )
            try:
                article = scrape_article(president["link"], subpage)
                output_path = output_dir / filename
                output_path.write_text(article["text"])
            except Exception as e:
                print(f"  ERROR: {e}")
            time.sleep(REQUEST_DELAY_SECONDS)


if __name__ == "__main__":
    main()
