import os
import xml.etree.ElementTree as ET

os.chdir(r"C:\Users\santo\OneDrive\Desktop\Data science\Feb 2026\2nd feb")

tree = ET.parse("769952.xml")
root = tree.getroot()

print(root)

root=ET.tostring(root, encoding='utf8').decode('utf8')
root

from bs4 import BeautifulSoup
import re

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub(r'\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = re.sub(' ', '', text)
    return text

sample = denoise_text(root)



