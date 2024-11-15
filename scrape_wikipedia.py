import requests
from bs4 import BeautifulSoup

from people import PRESIDENTS, SECRETARIES_OF_STATE

BASE_URL = "https://en.wikipedia.org/w/api.php"
DESTINATION_FOLDER = "text"


def get_pagetitle(query: str) -> str:
    """
    Return pagetitle of article that best matches a given query string
    See https://www.mediawiki.org/wiki/API:Opensearch
    """
    payload = {
        "action": "opensearch",
        "search": query,
        "limit": "5",
        "format": "json",
    }
    candidate_titles = requests.get(BASE_URL, params=payload).json()[1]
    if len(candidate_titles) > 0:
        first_result = candidate_titles[0]
        return first_result


def get_section_indices(pagetitle: str) -> str:
    """
    Return sections of article with a given pagetitle
    See https://www.mediawiki.org/wiki/API:Parsing_wikitext
    """
    disallowed_section_titles = [
        "See also",
        "Notes",
        "References",
        "Works cited",
        "Further reading",
        "External links",
        "Bibliography",
    ]
    payload = {
        "action": "parse",
        "page": pagetitle,
        "prop": "sections",
        "format": "json",
    }
    response = requests.get(BASE_URL, params=payload).json()
    section_indices = [
        section["index"]
        for section in response["parse"]["sections"]
        if section["level"] == "2" and section["line"] not in disallowed_section_titles
    ]
    return section_indices


def process_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove unwanted html elements"""
    for table in soup.find_all("table"):
        table.decompose()
    for infobox in soup.find_all("infobox"):
        infobox.decompose()
    for figcaption in soup.find_all("figcaption"):
        figcaption.decompose()
    for gallerytext in soup.find_all(class_="gallerytext"):
        gallerytext.decompose()
    for thumbcaption in soup.find_all(class_="thumbcaption"):
        thumbcaption.decompose()
    return soup


def process_text(raw_text: str) -> str:
    lines = [
        line
        for line in raw_text.splitlines()
        if (
            not line.startswith("^")  # Remove citation footnotes
            and "Cite error"
            not in line  # Remove errors in parsing particular site elements
        )
    ]
    lines = list(
        dict.fromkeys(lines)
    )  # Remove duplicates (e.g. multiple newline chars) but retain order
    clean_text = "\n".join(lines).strip("\n")
    return clean_text


def get_text(
    pagetitle: str,
    section_indices: list[str],
) -> str:
    """
    Return text from given section indices of article with given pagetitle.
    See https://www.mediawiki.org/wiki/API:Parsing_wikitext
    """
    sections = list()
    for section_index in section_indices:
        payload = {
            "action": "parse",
            "page": pagetitle,
            "prop": "text",
            "section": section_index,
            "disabletoc": True,
            "disableeditsection": True,
            "format": "json",
        }
        response = requests.get(BASE_URL, params=payload).json()
        soup_raw = BeautifulSoup(response["parse"]["text"]["*"], features="lxml")
        soup_clean = process_soup(soup_raw)
        text_raw = soup_clean.get_text()
        text_clean = process_text(text_raw)
        sections.append(text_clean)
    full_article = "\n".join(sections)
    return full_article


def scrape_wiki_article(query_title: str) -> str:
    """Return text of article that best matches a given query string"""
    pagetitle = get_pagetitle(query_title)
    if pagetitle is not None:
        section_indices = get_section_indices(pagetitle)
        text = get_text(pagetitle, section_indices)
        return text


def dump_to_txt(
    string: str,
    filename: str,
) -> None:
    """Dump a string into a .txt file"""
    filename_clean = filename.replace(" ", "_").replace(".", "").lower()
    with open(f"{DESTINATION_FOLDER}/{filename_clean}.txt", "w") as handle:
        handle.write(string)


if __name__ == "__main__":
    people = sorted(list(set(PRESIDENTS + SECRETARIES_OF_STATE)))
    for i, person in enumerate(people[:3]):
        print(f"[{i+1:3}/{len(people)}] Scraping... {person}")
        text = scrape_wiki_article(person)
        dump_to_txt(text, person)
