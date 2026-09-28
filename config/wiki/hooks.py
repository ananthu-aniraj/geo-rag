import re


def on_page_markdown(markdown, page, config, files):
    """
    Hook to dynamically adapt index.md (which is a symlink to README.md at root)
    for the MkDocs website.
    """
    if page.file.src_path == "index.md":
        # 1. Strip explicit mkdocs:hide comment blocks if present
        markdown = re.sub(
            r"<!--\s*mkdocs:hide\s*-->.*?<!--\s*/mkdocs:hide\s*-->",
            "",
            markdown,
            flags=re.DOTALL,
        )

        # 2. Remove the interactive wiki badge (circular link to the docs site itself)
        markdown = re.sub(r"\[!\[Wiki / Docs\]\([^)]+\)\]\([^)]+\)\s*", "", markdown)

        # 3. Remove the self-referential docs link and "direct browsing within repository" phrasing
        markdown = re.sub(
            r"Visit our full interactive documentation and pipeline walkthroughs at:\s*\n+"
            r"👉\s*\*\*\[Geo-RAG Interactive Wiki & Docs\]\([^)]+\)\*\*\s*\n+"
            r"For direct browsing within this repository:\s*\n+",
            "",
            markdown,
        )

        # 4. Replace markdown links like [Text](docs/path) with [Text](path) for MkDocs resolution
        markdown = markdown.replace("](docs/", "](")

        # 5. Rewrite root scripts links to point to the GitHub repository rather than broken local docs links
        markdown = re.sub(
            r"\]\(scripts/",
            "](https://github.com/ananthu-aniraj/geo-rag/blob/main/scripts/",
            markdown,
        )

    return markdown
