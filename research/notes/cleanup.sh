#!/bin/bash

find . -type f -name "*.aux" | xargs rm
find . -type f -name "*.fdb_latexmk" | xargs rm
find . -type f -name "*.fls" | xargs rm
find . -type f -name "*.log" | xargs rm
find . -type f -name "*.synctex.gz" | xargs rm
find . -type f -name "*.out" | xargs rm
find . -type f -name "*.bbl" | xargs rm
find . -type f -name "*.bcf" | xargs rm
find . -type f -name "*.blg" | xargs rm
find . -type f -name "*.run.xml" | xargs rm