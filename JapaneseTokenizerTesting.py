from transformers import BertJapaneseTokenizer

# Encode #

trainEntries = input("Enter the text to tokenize\n")

max_length = 512  # Maximum sequence length for BERT
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', max_len=max_length)

trainTokens = []
train_text_segments = [trainEntries[i:i+max_length] for i in range(0, len(trainEntries), max_length)]
for segment in train_text_segments:
    train_segment_tokens = tokenizer.encode(segment, add_special_tokens=True)
    trainTokens.extend(train_segment_tokens)

print(trainTokens)

# Decode #

trainEntries = input("Enter the text to de-tokenize\n")

token_ids = [int(x) for x in trainEntries.strip("[]").split(", ")]
decoded_text = tokenizer.decode(token_ids)

print(decoded_text)