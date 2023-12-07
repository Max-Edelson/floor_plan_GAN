#!/bin/bash

data='data/cubicasa5k/'
sets=('colorful' 'high_quality' 'high_quality_architectural')
targetPath="${data}svgs/"

if [ -d "$targetPath" ]; then
    rm -rf "$targetPath"
fi
mkdir -p "$targetPath"

for set in ${sets[@]}; do
    path="${data}${set}"
    i=1
    for dir in $path/*; do
        for f in $dir/*; do
            if echo $f | grep '.svg'; then
                cp "$f" "$targetPath"${set}_$i.svg
                i=$((i+1))
            fi
        do
    done
done