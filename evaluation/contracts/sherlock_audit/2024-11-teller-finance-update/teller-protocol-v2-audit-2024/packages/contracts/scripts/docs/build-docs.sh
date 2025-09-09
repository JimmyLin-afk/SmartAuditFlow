
#!/usr/bin/env bash
forge doc 

# replace the readme file 
rm docs/src/README.md
cp scripts/docs/welcome.md docs/src/README2.md

# replace the css 
rm docs/book.css
cp scripts/docs/book.css docs/book.css

forge doc --build