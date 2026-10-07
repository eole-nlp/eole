"""Stage website Markdown without changing repository documentation sources."""

import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DEST = DOCS / "build" / "site-docs"
GITHUB = "https://github.com/eole-nlp/eole/blob/main/"


def main():
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    sources = {}
    originals = {}

    def add(source, target, original=None):
        source = source.resolve()
        target = DEST / target
        sources[source] = target
        originals[source] = (original or source).resolve()

    # Handwritten site pages only; exclude generated symlink trees.
    site = DOCS / "docusaurus_tsx" / "docs"
    for pattern in ["*.md", "FAQ/*.md", "concepts/*.md"]:
        for source in site.glob(pattern):
            if not source.is_symlink():
                add(source, source.relative_to(site))
    add(ROOT / "README.md", "index.md")
    add(ROOT / "CONTRIBUTING.md", "contributing.md")
    # Include tracked recipe docs, not local scratch files or model artifacts.
    tracked = subprocess.check_output(["git", "ls-files", "recipes"], cwd=ROOT, text=True).splitlines()
    for name in tracked:
        source = ROOT / name
        if source.suffix == ".md":
            add(source, name)
    for source in DOCS.glob("*.md"):
        add(source, Path("guides") / source.name)
    reference = DOCS / "sphinx_markdown" / "markdown"
    for source in reference.rglob("*.md"):
        relative = source.relative_to(reference)
        original = DOCS / "source" / relative
        if not original.exists():
            original = original.with_suffix(".rst")
        add(source, Path("reference") / relative, original)
    for source, target in sources.items():
        text = source.read_text()

        def rewrite(match):
            label, href = match.groups()
            if href.startswith(("http:", "https:", "mailto:", "#", "/")):
                return match.group(0)
            path, separator, anchor = href.partition("#")
            resolved = (source.parent / unquote(path)).resolve()
            # Directory links in recipe READMEs point to their index page.
            if resolved.is_dir() and (resolved / "README.md").exists():
                resolved = (resolved / "README.md").resolve()
            if resolved in sources:
                href = Path(os.path.relpath(sources[resolved], target.parent)).as_posix()
                return f"[{label}]({href}{separator}{anchor})"
            if resolved.exists() and resolved.is_relative_to(ROOT):
                return f"[{label}]({GITHUB}{resolved.relative_to(ROOT).as_posix()}{separator}{anchor})"
            return match.group(0)

        # Preserve generated API HTML while putting block boundaries on new lines.
        text = text.replace("</summary>", "</summary>\n").replace("</p>", "</p>\n")
        text = text.replace("<s>", "&lt;s&gt;").replace("</s>", "&lt;/s&gt;")
        text = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", rewrite, text)
        original = originals[source]
        if original.exists() and original.is_relative_to(ROOT):
            edit_url = GITHUB.replace("/blob/", "/edit/") + original.relative_to(ROOT).as_posix()
            if text.startswith("---\n"):
                text = "---\ncustom_edit_url: " + edit_url + "\n" + text[4:]
            else:
                text = "---\ncustom_edit_url: " + edit_url + "\n---\n\n" + text
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    print(f"Staged {len(sources)} documentation pages in {DEST}")


if __name__ == "__main__":
    main()
