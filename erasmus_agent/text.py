"""Turn saved HTML pages into clean, readable text."""
from bs4 import BeautifulSoup

# Page furniture that never contains programme facts.
NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "form", "svg"]


def html_to_text(html: str) -> str:
    """Remove scripts, menus and footers, and return the text with one block per line."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    lines = (line.strip() for line in soup.get_text("\n").splitlines())
    return "\n".join(line for line in lines if line)