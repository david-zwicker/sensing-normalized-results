#!/usr/bin/env bash


sfood ../../*_response --internal |\
    grep -v "tests.py" > dependencies.txt

sfood-graph < dependencies.txt | \
    dot -Tps | pstopdf -i

mv stdin.pdf dependencies.pdf

cd ../..
pyreverse-2.7 -o pdf -p "" --ignore="tests.py" *_response

mv classes.pdf docs/dependencies
mv packages.pdf docs/dependencies
