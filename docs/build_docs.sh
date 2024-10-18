# Helper Script to build and assemble all docs components

# Components:
# - docusaurus_tsx = main website that will be served (self-host/AWS/github pages)
# - recipes = recipes folder containing some code as well as some readme.md which will be rendered in the doc
# - sphinx source = API reference doc automatically built with sphinx and rendered into markdown to add to main docusaurus

# 1 - build sphinx
git config --global --add safe.directory /work # patch for github ext to work
# build biblio
pandoc -t markdown_strict --citeproc source/ref.md -o source/bibliography.md --bibliography source/refs.bib
sphinx-build -M markdown ./source ./sphinx_markdown

cd sphinx_markdown/markdown
# ensure proper newlines for markdown render
find . -name "*.md" -print0 | xargs -0 sed -i 's/<\/summary>/<\/summary>\n/g'
find . -name "*.md" -print0 | xargs -0 sed -i 's/<\/p>/<\/p>\n/g'
# escape some conflicting html tags for proper render
find . -name "*.md" -print0 | xargs -0 sed -i 's/<s>/\&lt;s\&gt;/g'
find . -name "*.md" -print0 | xargs -0 sed -i 's/<\/s>/\&lt;\/s\&gt;/g'
[ -f ref.md ] && rm ref.md # only used to build bibliography.md with pandoc

cd ../..

# 2 - link sphinx markdown build to proper place
cd docusaurus_tsx/docs
[ -L reference ] && rm reference; ln -s ../../sphinx_markdown/markdown reference
cd ../..

# 3 - link recipes directory to proper place
cd docusaurus_tsx/docs
[ -L recipes ] && rm recipes; ln -s ../../../recipes
[ -f index.md ] && rm index.md; ln -s ../../../README.md index.md
[ -f contributing.md ] && rm contributing.md; ln -s ../../../CONTRIBUTING.md contributing.md

cd ../..

# 4 - build docusaurus
cd docusaurus_tsx
yarn clear && yarn build