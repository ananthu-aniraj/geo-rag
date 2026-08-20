import re

def on_page_markdown(markdown, page, config, files):
    """
    Hook to dynamically rewrite links pointing to docs/ inside index.md
    (which is a symlink to README.md at the root).
    """
    if page.file.src_path == 'index.md':
        # Replace markdown links like [Text](docs/path) with [Text](path)
        markdown = markdown.replace('](docs/', '](')
    return markdown
