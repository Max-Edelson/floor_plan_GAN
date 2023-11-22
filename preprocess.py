import re
import os

# Run once over all SVG
# collapse whole file into 1 line with no excess spacing and regex removal of properties. 
def translate_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = ' '.join(lines)
    lines = re.sub(r'<text.*?/text>', '', lines, flags=re.DOTALL)
    lines = re.sub(r'<desc.*?/desc>', '', lines, flags=re.DOTALL)
    lines = re.sub(r'(class|id)=".*?"', '', lines, flags=re.DOTALL)
    lines = re.sub(r'\s+', ' ', lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(lines)

path = 'data'
out = 'out'
for file in os.listdir(path):
    translate_file(f'{path}{os.sep}{file}', f'{out}{os.sep}{file}')
