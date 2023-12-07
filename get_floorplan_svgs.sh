#!/bin/bash

data='data/floorplan/'
sets=('test-00' 'train-00' 'train-01')
targetPath="${data}svgs/"

if [ -d "$targetPath" ]; then
    rm -rf "$targetPath"
fi
mkdir -p "$targetPath"

for set in ${sets[@]}; do
    path="${data}${set}"
    for f in $path/*; do
        if echo $f | grep '.svg'; then
            cp "$f" "$targetPath"
        fi
    done
done