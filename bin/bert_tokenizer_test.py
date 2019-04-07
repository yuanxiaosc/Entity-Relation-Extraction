import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../bert")))
import tokenization


def get_vocab_file_path(vocab_file_path):
    vocab_file_path = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../pretrained_model/chinese_L-12_H-768_A-12")),
        vocab_file_path)
    return vocab_file_path

vocab_file_path = "vocab.txt"

# BERT 自带WordPiece分词工具，对于中文都是分成单字
bert_tokenizer = tokenization.FullTokenizer(vocab_file=get_vocab_file_path(vocab_file_path), do_lower_case=True)  # 初始化 bert_token 工具

print(bert_tokenizer.convert_tokens_to_ids(["[SEP]"]))