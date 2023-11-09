from deep_translator import GoogleTranslator as Translator
import re
import os

# Run once over all SVG

def translate_file(input_file, output_file):
    translator = Translator()

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            chinese_phrases = re.findall(r'[\u4e00-\u9fff]+', line)
            for phrase in chinese_phrases:
                translation = translator.translate(phrase, source='zh-cn', target='en')
                if 'layer' in line:
                    translation = translation.upper().replace(' ', '_')
                line = line.replace(phrase, translation)
            f.write(line)

path = 'data'
for file in os.listdir(path):
    translate_file(f'{path}{os.sep}{file}', f'{path}{os.sep}{file}')