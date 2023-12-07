#!/bin/bash

data='data/cubicasa5k/'
sets=('colorful' 'high_quality' 'high_quality_architectural')
targetPath="${data}pngs/"

if [ -d "$targetPath" ]; then
    rm -rf "$targetPath"
fi
mkdir -p "$targetPath"

for set in ${sets[@]}; do
    path="${data}${set}"
    i=1
    for dir in $path/*; do
        for f in $dir/*; do
            if echo $f | grep '.png'; then
                cp "$f" "$targetPath"${set}_$i.png
                i=$((i+1))
            fi
        do
    done
done