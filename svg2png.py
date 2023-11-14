# `sudo apt-get install imagemagick` for ubuntu
# `brew install imagemagick` for macos

import os
import subprocess

def translate_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_text = False
    with open('temp.svg', 'w', encoding='utf-8') as f:
        for line in lines:
            if not in_text:
                in_text = '<text' in line
            if in_text:
                in_text = 'text>' not in line
                continue
            f.write(line)
    command = ['convert', '-resize', '256x256', 'temp.svg', output_file]
    subprocess.run(command, check=True)

in_path = 'data'
out_path = 'out'


for file in os.listdir(in_path):
    translate_file(f'{in_path}{os.sep}{file}', f'{out_path}{os.sep}{file}')
    break
os.remove('temp.svg')