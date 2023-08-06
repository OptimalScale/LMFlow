# Vocab Extension
## Train & Merge Tokenizer
To automatically convert data, train a SentencePiece tokenizer, and merge the tokenizer, you can run the following script:
```
bash scripts/vocab_extension/train_merge_tokenizer.sh
``` 
Alternatively, you can run each of the three steps separately:

## Convert JSON Data to TXT
To convert JSON data to TXT for sentencepiece tokenizer training, run:
```
bash scripts/vocab_extension/convert_json_to_txt.sh
```
## Train SentencePiece Tokenizer
To train a SentencePiece tokenizer, run:
```
bash scripts/vocab_extension/train_tokenizer.sh
```
## Merge New Tokenizer with the Origin One
To merge a new tokenizer with the original one, run:
```
bash scripts/vocab_extension/merge_tokenizer.sh
```