import os
import re


def on_page_markdown(markdown, page, config, files):
    """
    Hook to dynamically adapt index.md (which is a symlink to README.md at root)
    and pages linking to README.md for the MkDocs website.
    """
    # 0. Rewrite links to root README.md into index.md for MkDocs
    page_dir = os.path.dirname(page.file.src_path)
    if page_dir:
        rel_to_root = os.path.relpath(".", page_dir).replace("\\", "/") + "/index.md"
    else:
        rel_to_root = "index.md"

    def _replace_readme_link(match):
        anchor = match.group(1)
        if anchor:
            clean_anchor = anchor.lstrip("#-")
            return f"]({rel_to_root}#{clean_anchor})"
        return f"]({rel_to_root})"

    markdown = re.sub(
        r"\]\(\.\./README\.md(#[-a-zA-Z0-9_]*)?\)",
        _replace_readme_link,
        markdown,
    )

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
