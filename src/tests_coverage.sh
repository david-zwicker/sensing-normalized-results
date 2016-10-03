#!/bin/bash
echo 'Determine coverage of all unittests...'

mkdir -p docs/coverage

nosetests-2.7 --with-coverage \
    --cover-erase --cover-inclusive \
    --cover-package=binary_response,adaptive_response \
    --cover-html --cover-html-dir="docs/coverage"
