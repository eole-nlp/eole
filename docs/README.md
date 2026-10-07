# Build and maintain the documentation

The website combines three sources:

- `docusaurus_tsx/docs`: handwritten quickstart, concepts, and FAQs.
- `source`: Sphinx configuration/API references generated from Python objects.
- Repository README, contributing guide, recipes, and `docs/*.md` guides.

Use a Python environment with Eole's dependencies, Node.js 20 or newer, and
Corepack. Run from the repository root:

```bash
python -m pip install -r docs/requirements.txt
(cd docs/docusaurus_tsx && corepack yarn install --immutable)
bash docs/build_docs.sh
```

The script builds Sphinx Markdown, stages pages in `docs/build/site-docs`, checks
TypeScript, and builds `docs/docusaurus_tsx/build`. It does not publish the site.
For a standalone HTML API reference, run from the repository root:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python -m sphinx -E -a -W -b html docs/source docs/build/html
```

Edit the source files, not `build/`, `sphinx_markdown/`, or staged pages. Site
assembly keeps recipe links to other documentation within the site and links
code/config assets to GitHub. Each generated page has an edit link to its real
source. Bibliography and Mermaid nodes are handled by the Sphinx Markdown
translator in `source/conf.py`.

The deployment workflow uses the Yarn lockfile. Keep Docusaurus packages aligned
and update the lockfile when changing site dependencies. The generated site
checks broken page links and anchors; this does not prove that external URLs, full training,
or GPU model examples execute successfully.
