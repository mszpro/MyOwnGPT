import re #regux正規表現モジュール
import numpy as np
import json # convert data result to json entries
import os
import random
from transformers import BertJapaneseTokenizer

def read_markdown_file(file_path):
    if file_path.endswith(".DS_Store"):
        return ""
    with open(file_path, mode="r", encoding="utf-8") as md_file:
        print("Reading", file_path)
        content = md_file.read()
    return content

dataSetFileNames = os.listdir(os.path.join(os.path.dirname(__file__), 'dataset'))
dataSetFileNames = [file for file in dataSetFileNames if file != '.DS_Store']

if len(dataSetFileNames) < 1:
    print("No files in the sample data directory. Put some text files or .md files there!")
    exit()

print("Reading file names", dataSetFileNames)

def remove_code_blocks(markdown_text):
    # Remove fenced code blocks
    fenced_code_blocks = re.compile(r'```[\s\S]*?```')
    markdown_text = re.sub(fenced_code_blocks, '', markdown_text)

    # Remove inline code blocks
    inline_code_blocks = re.compile(r'`[^`]*`')
    markdown_text = re.sub(inline_code_blocks, '', markdown_text)

    return markdown_text

entries = ""
for filename in dataSetFileNames:
    file_path = os.path.join(os.path.dirname(__file__), 'dataset', filename)
    print("Processing ", file_path)
    markdown_content = read_markdown_file(file_path)
    markdown_content = remove_code_blocks(markdown_content) # remove the code blocks
    markdown_content = re.sub(r"!\[.*?\]\(.*?\)", "", markdown_content) # Remove ![] image tag
    markdown_content = re.sub(r"<img.*?>", "", markdown_content) # Remove image tags
    markdown_content = re.sub(r"\[.*?\]\(.*?\)", "", markdown_content) # Remove markdown link tags
    markdown_content = re.sub(r"http\S+", "", markdown_content) # remove URL
    for line in markdown_content.split("\n"):
        if len(line) > 5:
            entries += line + "\n\n"

n = len(entries)
print("Total input entry length", n)

trainEntries = entries[:int(n*0.9)]
evalEntries = entries[int(n*0.9):]

print("Train entries count:", len(trainEntries), " eval entries count ", len(evalEntries))

max_length = 512  # Maximum sequence length for BERT
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', max_len=max_length)

trainTokens = []
train_text_segments = [trainEntries[i:i+max_length] for i in range(0, len(trainEntries), max_length)]
for segment in train_text_segments:
    train_segment_tokens = tokenizer.encode(segment, add_special_tokens=True)
    trainTokens.extend(train_segment_tokens)

evalTokens = []
eval_text_segments = [evalEntries[i:i+max_length] for i in range(0, len(evalEntries), max_length)]
for segment in eval_text_segments:
    eval_segment_tokens = tokenizer.encode(segment, add_special_tokens=True)
    evalTokens.extend(eval_segment_tokens)

print(len(trainTokens), "used for training;", len(evalTokens), "used for eval")

train_ids = np.array(trainTokens)
val_ids = np.array(evalTokens)
newFolderPath = os.path.join(os.path.dirname(__file__), 'TrainingSet')
if not os.path.exists(newFolderPath):
    os.makedirs(newFolderPath)
train_ids.tofile(os.path.join(newFolderPath, 'train.bin'))
val_ids.tofile(os.path.join(newFolderPath, 'val.bin'))