# coding=utf-8
import os
import json

# 获取最新模型预测数据文件夹
def get_latest_model_predict_data_dir(new_epochs_ckpt_dir=None):
    # 获取文件下最新文件路径
    def new_report(test_report):
        lists = os.listdir(test_report)  # 列出目录的下所有文件和文件夹保存到lists
        lists.sort(key=lambda fn: os.path.getmtime(test_report + "/" + fn))  # 按时间排序
        file_new = os.path.join(test_report, lists[-1])  # 获取最新的文件保存到file_new
        return file_new
    if new_epochs_ckpt_dir is None:
        # 获取分类预测输出文件路径
        input_new_epochs = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "output")), "sequnce_infer_out")
        # 获取最新周期文件路径
        new_ckpt_dir = new_report(input_new_epochs)
        input_new_epochs_ckpt = os.path.join(input_new_epochs, new_ckpt_dir)
        # 获取最新周期下最新模型文件路径
        new_epochs_ckpt_dir = new_report(input_new_epochs_ckpt)
    if not os.path.exists(new_ckpt_dir):
        raise ValueError("路径不存在！{}".format(new_epochs_ckpt_dir))
    return new_epochs_ckpt_dir

# dict is comes from raw_data all_50_schemas
schemas_dict_relation_2_object_subject_type = {
    '父亲': [('人物', '人物')],
    '妻子': [('人物', '人物')],
    '母亲': [('人物', '人物')],
    '丈夫': [('人物', '人物')],
    '祖籍': [('地点', '人物')],
    '总部地点': [('地点', '企业')],
    '出生地': [('地点', '人物')],
    '目': [('目', '生物')],
    '面积': [('Number', '行政区')],
    '简称': [('Text', '机构')],
    '上映时间': [('Date', '影视作品')],
    '所属专辑': [('音乐专辑', '歌曲')],
    '注册资本': [('Number', '企业')],
    '首都': [('城市', '国家')],
    '导演': [('人物', '影视作品')],
    '字': [('Text', '历史人物')],
    '身高': [('Number', '人物')],
    '出品公司': [('企业', '影视作品')],
    '修业年限': [('Number', '学科专业')],
    '出生日期': [('Date', '人物')],
    '制片人': [('人物', '影视作品')],
    '编剧': [('人物', '影视作品')],
    '国籍': [('国家', '人物')],
    '海拔': [('Number', '地点')],
    '连载网站': [('网站', '网络小说')],
    '朝代': [('Text', '历史人物')],
    '民族': [('Text', '人物')],
    '号': [('Text', '历史人物')],
    '出版社': [('出版社', '书籍')],
    '主持人': [('人物', '电视综艺')],
    '专业代码': [('Text', '学科专业')],
    '歌手': [('人物', '歌曲')],
    '作词': [('人物', '歌曲')],
    '主角': [('人物', '网络小说')],
    '董事长': [('人物', '企业')],
    '成立日期': [('Date', '机构'), ('Date', '企业')],
    '毕业院校': [('学校', '人物')],
    '占地面积': [('Number', '机构')],
    '官方语言': [('语言', '国家')],
    '邮政编码': [('Text', '行政区')],
    '人口数量': [('Number', '行政区')],
    '所在城市': [('城市', '景点')],
    '作者': [('人物', '图书作品')],
    '作曲': [('人物', '歌曲')],
    '气候': [('气候', '行政区')],
    '嘉宾': [('人物', '电视综艺')],
    '主演': [('人物', '影视作品')],
    '改编自': [('作品', '影视作品')],
    '创始人': [('人物', '企业')]}

class File_Management(object):
    """读取TXT文件，以列表形式返回文件内容"""
    def __init__(self, TEST_DATA_DIR=None, MODEL_OUTPUT_DIR=None, Competition_Mode=True):
        self.TEST_DATA_DIR = TEST_DATA_DIR
        self.MODEL_OUTPUT_DIR = get_latest_model_predict_data_dir(MODEL_OUTPUT_DIR)
        self.Competition_Mode = Competition_Mode

    def file_path_and_name(self):
        text_sentence_file_path = os.path.join(self.TEST_DATA_DIR, "text_and_one_predicate.txt")
        token_in_file_path = os.path.join(self.TEST_DATA_DIR, "token_in_not_UNK_and_one_predicate.txt")
        predicate_token_label_file_path = os.path.join(self.MODEL_OUTPUT_DIR, "token_label_predictions.txt")

        file_path_list = [text_sentence_file_path, token_in_file_path, predicate_token_label_file_path]
        file_name_list = ["text_sentence_list", "token_in_not_NUK_list ", "token_label_list",]
        if not self.Competition_Mode:
            spo_out_file_path = os.path.join(self.TEST_DATA_DIR, "spo_out.txt")
            if os.path.exists(spo_out_file_path):
                file_path_list.append(spo_out_file_path)
                file_name_list.append("reference_spo_list")
        return file_path_list, file_name_list

    def read_file_return_content_list(self):
        file_path_list, file_name_list = self.file_path_and_name()
        content_list_summary = []
        for file_path in file_path_list:
            with open(file_path, "r", encoding='utf-8') as f:
                content_list = f.readlines()
                content_list = [content.replace("\n", "") for content in content_list]
                content_list_summary.append(content_list)

        if self.Competition_Mode:
            content_list_length_summary = [(file_name, len(content_list)) for content_list, file_name in
                                           zip(content_list_summary, file_name_list)]
            file_line_number = self._check_file_line_numbers(content_list_length_summary)
            print("Competition_Mode=True, check file line pass!")
            print("输入文件行数一致，行数是: ", file_line_number)
        else:
            file_line_number = len(content_list_summary[0])
            print("first file line number: ", file_line_number)
            print("do not check file line! if you need check file line, set Competition_Mode=True")
        print("\n")
        return content_list_summary, file_line_number

    def _check_file_line_numbers(self, content_list_length_summary):
        content_list_length_file_one = content_list_length_summary[0][1]
        for file_name, file_line_number in content_list_length_summary:
            assert file_line_number == content_list_length_file_one
        return content_list_length_file_one


class Sorted_relation_and_entity_list_Management(File_Management):
    """
    生成按概率大小排序的可能关系列表和按照原始句子中顺序排序的实体列表
    """
    def __init__(self, TEST_DATA_DIR, MODEL_OUTPUT_DIR, Competition_Mode=False):
        File_Management.__init__(self, TEST_DATA_DIR=TEST_DATA_DIR, MODEL_OUTPUT_DIR=MODEL_OUTPUT_DIR, Competition_Mode=Competition_Mode)
        # 关系列表 把模型输出的实数值对应为标签
        self.relationship_label_list = ['丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地', '出生日期', '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑', '改编自', '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高', '连载网站', '邮政编码', '面积', '首都']
        self.Competition_Mode = Competition_Mode
        print("test数据输入路径是:\t{}".format(self.TEST_DATA_DIR))
        print("最新模型预测结果路径是:\t{}".format(self.MODEL_OUTPUT_DIR))

    def get_input_list(self,):
        content_list_summary, self.file_line_number = self.read_file_return_content_list()
        if len(content_list_summary) == 4:
            [text_sentence_list, token_in_not_NUK_list, token_label_list, reference_spo_list] = content_list_summary
        elif len(content_list_summary) == 3:
            [text_sentence_list, token_in_not_NUK_list, token_label_list] = content_list_summary
            reference_spo_list = [None] * len(text_sentence_list)
        else:
            raise ValueError("check code!")
        return text_sentence_list, token_in_not_NUK_list, token_label_list, reference_spo_list


    #合并由WordPiece切分的词和单字
    def _merge_WordPiece_and_single_word(self, entity_sort_list):
        # [..['B-SUB', '新', '地', '球', 'ge', '##nes', '##is'] ..]---> [..('SUB', '新地球genesis')..]
        entity_sort_tuple_list = []
        for a_entity_list in entity_sort_list:
            entity_content = ""
            entity_type = None
            for idx, entity_part in enumerate(a_entity_list):
                if idx == 0:
                    entity_type = entity_part
                    if entity_type[:2] not in ["B-", "I-"]:
                        break
                else:
                    if entity_part.startswith("##"):
                        entity_content += entity_part.replace("##", "")
                    else:
                        entity_content += entity_part
            if entity_content != "":
                entity_sort_tuple_list.append((entity_type[2:], entity_content))
        return entity_sort_tuple_list

    # 把spo_out.txt 的[SPO_SEP] 分割形式转换成标准列表字典形式
    # 妻子 人物 人物 杨淑慧 周佛海[SPO_SEP]丈夫 人物 人物 周佛海 杨淑慧 ---> dict
    def preprocessing_reference_spo_list(self, refer_spo_str):
        refer_spo_list = refer_spo_str.split("[SPO_SEP]")
        refer_spo_list = [spo.split(" ") for spo in refer_spo_list]
        refer_spo_list = [dict([('predicate', spo[0]),
                                ('object_type', spo[2]), ('subject_type', spo[1]),
                                ('object', spo[4]), ('subject', spo[3])]) for spo in refer_spo_list]
        refer_spo_list.sort(key= lambda item:item['predicate'])
        return refer_spo_list

    # 把模型输出实体标签按照原句中相对位置输出
    def model_token_label_2_entity_sort_tuple_list(self, token_in_not_UNK_list, predicate_token_label_list):
        """
        :param token_in_not_UNK:  ['紫', '菊', '花', '草', '是', '菊', '目', '，', '菊', '科', '，', '松', '果', '菊', '属', '的', '植', '物']
        :param predicate_token_label: ['B-SUB', 'I-SUB', 'I-SUB', 'I-SUB', 'O', 'B-OBJ', 'I-OBJ', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        :return: [('SUB', '紫菊花草'), ('OBJ', '菊目')]
        """
        # 除去模型输出的特殊符号
        def preprocessing_model_token_lable(predicate_token_label_list, token_in_list_lenth):
            # ToDo:检查错误，纠错
            if predicate_token_label_list[0] == "[CLS]":
                predicate_token_label_list = predicate_token_label_list[1:]  # y_predict.remove('[CLS]')
            if len(predicate_token_label_list) > token_in_list_lenth:  # 只取输入序列长度即可
                predicate_token_label_list = predicate_token_label_list[:token_in_list_lenth]
            return predicate_token_label_list
        # 预处理标注数据列表
        predicate_token_label_list = preprocessing_model_token_lable(predicate_token_label_list, len(token_in_not_UNK_list))
        entity_sort_list = []
        entity_part_list = []
        #TODO:需要检查以下的逻辑判断，可能写的不够完备充分
        for idx, token_label in enumerate(predicate_token_label_list):
            # 如果标签为 "O"
            if token_label == "O":
                # entity_part_list 不为空，则直接提交
                if len(entity_part_list) > 0:
                    entity_sort_list.append(entity_part_list)
                    entity_part_list = []
            # 如果标签以字符 "B-" 开始
            if token_label.startswith("B-"):
                # 如果 entity_part_list 不为空，则先提交原来 entity_part_list
                if len(entity_part_list) > 0:
                    entity_sort_list.append(entity_part_list)
                    entity_part_list = []
                entity_part_list.append(token_label)
                entity_part_list.append(token_in_not_UNK_list[idx])
                # 如果到了标签序列最后一个标签处
                if idx == len(predicate_token_label_list) - 1:
                    entity_sort_list.append(entity_part_list)
            # 如果标签以字符 "I-"  开始 或者等于 "[##WordPiece]"
            if token_label.startswith("I-") or token_label == "[##WordPiece]":
                # entity_part_list 不为空，则把该标签对应的内容并入 entity_part_list
                if len(entity_part_list) > 0:
                    entity_part_list.append(token_in_not_UNK_list[idx])
                    # 如果到了标签序列最后一个标签处
                    if idx == len(predicate_token_label_list) - 1:
                        entity_sort_list.append(entity_part_list)
            # 如果遇到 [SEP] 分隔符，说明需要处理的标注部分已经结束
            if token_label == "[SEP]":
                break
        entity_sort_tuple_list = self._merge_WordPiece_and_single_word(entity_sort_list)
        return entity_sort_tuple_list

    # 生成排好序的关系列表和实体列表
    def produce_relationship_and_entity_sort_list(self):
        text_sentence_list, token_in_not_NUK_list, token_label_list, reference_spo_list = self.get_input_list()
        for [text_sentence, token_in_not_UNK, token_label, refer_spo_str] in\
                zip(text_sentence_list, token_in_not_NUK_list, token_label_list, reference_spo_list):
            text = text_sentence.split("\t")[0]
            text_predicate = text_sentence.split("\t")[1]
            token_in = token_in_not_UNK.split("\t")[0].split(" ")
            token_in_predicate = token_in_not_UNK.split("\t")[1]
            assert text_predicate == token_in_predicate
            token_label_out = token_label.split(" ")
            entity_sort_tuple_list = self.model_token_label_2_entity_sort_tuple_list(token_in, token_label_out)
            if self.Competition_Mode:
                yield text, text_predicate, entity_sort_tuple_list, None
            else:
                if refer_spo_str is not None:
                    refer_spo_list = self.preprocessing_reference_spo_list(refer_spo_str)
                else:
                    refer_spo_list = []
                yield text, text_predicate, entity_sort_tuple_list, refer_spo_list

    # 打印排好序的关系列表和实体列表
    def show_produce_relationship_and_entity_sort_list(self):
        idx = 0
        for text, text_predicate, entity_sort_tuple_list, refer_spo_list in self.produce_relationship_and_entity_sort_list():
            print("序号：           ", idx + 1)
            print("原句：           ", text)
            print("预测的关系：     ", text_predicate)
            print("预测的实体：     ", entity_sort_tuple_list)
            print("参考的 spo_slit：", refer_spo_list)
            print("\n")
            idx += 1
            if idx == 100:
                break

    def produce_output_file(self, OUT_RESULTS_DIR=None, keep_empty_spo_list=False):
        filename = "subject_predicate_object_predict_output.json"
        output_dict = dict()
        for text, text_predicate, entity_sort_tuple_list, refer_spo_list in self.produce_relationship_and_entity_sort_list():
            object_type, subject_type = schemas_dict_relation_2_object_subject_type[text_predicate][0]
            subject_list = [value for name, value in entity_sort_tuple_list if name == "SUB"]
            subject_list = list(set(subject_list))
            subject_list = [value for value in subject_list if len(value) >= 2]
            object_list = [value for name, value in entity_sort_tuple_list if name == "OBJ"]
            object_list = list(set(object_list))
            object_list = [value for value in object_list if len(value) >= 2]
            if len(subject_list) == 0 or len(object_list) == 0:
                output_dict.setdefault(text, [])
            for subject_value in subject_list:
                for object_value in object_list:
                    output_dict.setdefault(text, []).append({"object_type": object_type, "predicate": text_predicate,
                                                             "object": object_value, "subject_type": subject_type,
                                                             "subject": subject_value})
        if keep_empty_spo_list:
            filename = "keep_empty_spo_list_" + filename
        if OUT_RESULTS_DIR is None:
            out_path = filename
        else:
            out_path = os.path.join(OUT_RESULTS_DIR, filename)
        print("生成结果的输出路径是:\t{}".format(out_path))
        if not os.path.exists(OUT_RESULTS_DIR):
            os.makedirs(OUT_RESULTS_DIR)
        result_json_write_f = open(out_path, "w", encoding='utf-8')
        count_line_number = 0
        count_empty_line_number = 0
        for text, spo_list in output_dict.items():
            count_line_number += 1
            line_dict = dict()
            line_dict["text"] = text
            line_dict["spo_list"] = spo_list
            line_json = json.dumps(line_dict, ensure_ascii=False)
            if len(spo_list) == 0:
                count_empty_line_number += 1
            if keep_empty_spo_list:
                result_json_write_f.write(line_json + "\n")
            else:
                if len(spo_list) > 0:
                    result_json_write_f.write(line_json + "\n")
        print("empty_line: {}, line: {}, percentage: {:.2f}%".format(count_empty_line_number, count_line_number,
                                                                     (count_empty_line_number / count_line_number) * 100))


if __name__=='__main__':
    TEST_DATA_DIR = "bin/subject_object_labeling/sequence_labeling_data/test"
    # MODEL_OUTPUT_DIR = "output/sequnce_infer_out/epochs9/ckpt20000"
    MODEL_OUTPUT_DIR = None
    OUT_RESULTS_DIR = "output/final_text_spo_list_result"
    Competition_Mode = True
    spo_list_manager = Sorted_relation_and_entity_list_Management(TEST_DATA_DIR, MODEL_OUTPUT_DIR, Competition_Mode=Competition_Mode)
    spo_list_manager.produce_output_file(OUT_RESULTS_DIR=OUT_RESULTS_DIR, keep_empty_spo_list=True)