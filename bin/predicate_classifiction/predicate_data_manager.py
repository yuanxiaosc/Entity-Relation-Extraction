import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bert")))
import tokenization

print("if not have raw data, please dowload data from http://lic2019.ccf.org.cn/kg !")

def unzip_and_move_files():
    "解压原始文件并且放入 raw_data 文件夹下面"
    os.system("unzip dev_data.json.zip")
    os.system("mv dev_data.json raw_data/dev_data.json")
    os.system("unzip train_data.json.zip")
    os.system("mv train_data.json raw_data/train_data.json")


class Model_data_preparation(object):

    def __init__(self, DATA_INPUT_DIR="raw_data", DATA_OUTPUT_DIR="SKE_2019_tokened_labeling",
                 vocab_file_path="vocab.txt", do_lower_case=True, Competition_Mode=False):
        # BERT 自带WordPiece分词工具，对于中文都是分成单字
        self.bert_tokenizer = tokenization.FullTokenizer(vocab_file=self.get_vocab_file_path(vocab_file_path),
                                                         do_lower_case=do_lower_case)  # 初始化 bert_token 工具
        self.DATA_INPUT_DIR = self.get_data_input_dir(DATA_INPUT_DIR)
        self.DATA_OUTPUT_DIR = DATA_OUTPUT_DIR
        self.Competition_Mode = Competition_Mode

    def get_data_input_dir(self, DATA_INPUT_DIR):
        DATA_INPUT_DIR = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")), DATA_INPUT_DIR)
        return DATA_INPUT_DIR

    def get_vocab_file_path(self, vocab_file_path):
        vocab_file_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pretrained_model/chinese_L-12_H-768_A-12")), vocab_file_path)
        return vocab_file_path

    # 处理原始数据
    def separate_raw_data_and_token_labeling(self):
        if not os.path.exists(self.DATA_OUTPUT_DIR):
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "train"))
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "valid"))
            os.makedirs(os.path.join(self.DATA_OUTPUT_DIR, "test"))

        for file_set_type in ["train", "valid", "test"]:
            print(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type)))
            if file_set_type in ["train", "valid"]:
                predicate_out_f = open(
                    os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "predicate_out.txt"), "w",
                    encoding='utf-8')
            text_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "text.txt"), "w",
                          encoding='utf-8')
            token_in_f = open(os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "token_in.txt"), "w",
                              encoding='utf-8')
            token_in_not_UNK_f = open(
                os.path.join(os.path.join(self.DATA_OUTPUT_DIR, file_set_type), "token_in_not_UNK.txt"), "w",
                encoding='utf-8')

            #
            def predicate_to_predicate_file(spo_list):
                predicate_list = [spo['predicate'] for spo in spo_list]
                predicate_list_str = " ".join(predicate_list)
                predicate_out_f.write(predicate_list_str + "\n")

            if file_set_type == "train":
                path_to_raw_data_file = "train_data.json"
            elif file_set_type == "valid":
                path_to_raw_data_file = "dev_data.json"
            else:
                if self.Competition_Mode == True:
                    path_to_raw_data_file = "test1_data_postag.json"
                else:
                    path_to_raw_data_file = "dev_data.json"

            with open(os.path.join(self.DATA_INPUT_DIR, path_to_raw_data_file), 'r', encoding='utf-8') as f:
                count_numbers = 0
                while True:
                    line = f.readline()
                    if line:
                        count_numbers += 1
                        r = json.loads(line)
                        if file_set_type in ["train", "valid"]:
                            spo_list = r["spo_list"]
                        else:
                            spo_list = []
                        text = r["text"]
                        text_tokened = self.bert_tokenizer.tokenize(text)
                        text_tokened_not_UNK = self.bert_tokenizer.tokenize_not_UNK(text)

                        if file_set_type in ["train", "valid"]:
                            predicate_to_predicate_file(spo_list)
                        text_f.write(text + "\n")
                        token_in_f.write(" ".join(text_tokened) + "\n")
                        token_in_not_UNK_f.write(" ".join(text_tokened_not_UNK) + "\n")
                    else:
                        break
            print(file_set_type)
            print("all numbers", count_numbers)
            print("\n")
            text_f.close()
            token_in_f.close()
            token_in_not_UNK_f.close()
            if file_set_type in ["train", "valid"]:
                predicate_out_f.close()


if __name__ == "__main__":
    DATA_DIR = "raw_data"
    DATA_OUTPUT_DIR = "classfication_data"
    Competition_Mode = True
    model_data = Model_data_preparation(
        DATA_INPUT_DIR=DATA_DIR, DATA_OUTPUT_DIR=DATA_OUTPUT_DIR, Competition_Mode=Competition_Mode)
    model_data.separate_raw_data_and_token_labeling()
