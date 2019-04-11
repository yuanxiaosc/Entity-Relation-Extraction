import os

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
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output")), "predicate_infer_out")
        # 获取最新周期文件路径
        new_ckpt_dir = new_report(input_new_epochs)
        input_new_epochs_ckpt = os.path.join(input_new_epochs, new_ckpt_dir)
        # 获取最新周期下最新模型文件路径
        new_epochs_ckpt_dir = new_report(input_new_epochs_ckpt)
    return new_epochs_ckpt_dir

#对于没有预测出关系的句子，启发式地选择相对概率最大的10个关系作为输出
def replace_empty_infer_predicate_to_three_possible_values(predicate_score_value):
    label_list = ['丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地', '出生日期',
              '创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑', '改编自',
              '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高', '连载网站',
              '邮政编码', '面积', '首都']
    predicate_score_value_list = predicate_score_value.split(" ")
    predicate_score_name_value_list = [(label, value) for label, value in zip(label_list, predicate_score_value_list)]
    predicate_score_name_value_sort_list = sorted(predicate_score_name_value_list, key=lambda x: x[1], reverse=True)
    name_value_three_items = predicate_score_name_value_sort_list[:10]
    three_predicate_list = [name for name, value in name_value_three_items]
    return three_predicate_list


def prepare_data_for_subject_object_labeling_infer(predicate_classifiction_input_file_dir,
                                                   predicate_classifiction_infer_file_dir, out_file):
    """
    Converting the predicted results of the multi-label classification model
    into the input format required by the sequential label model
    :param predicate_classifiction_input_file_dir: Path of Input file of classification model
    :param predicate_classifiction_infer_file_dir: Path of Predictive Output of Classification Model
    :param out_file: Path of Input file of sequential labeling model
    :return: Input file of sequential labeling model
    """
    text_file = open(os.path.join(predicate_classifiction_input_file_dir, "text.txt"),
                     "r", encoding='utf-8').readlines()
    token_in_file = open(os.path.join(predicate_classifiction_input_file_dir, "token_in.txt"),
                         "r", encoding='utf-8').readlines()
    token_in_not_UNK_file = open(os.path.join(predicate_classifiction_input_file_dir, "token_in_not_UNK.txt"),
                         "r", encoding='utf-8').readlines()
    new_epochs_ckpt_dir = get_latest_model_predict_data_dir(predicate_classifiction_infer_file_dir)
    predicate_predict_file = open(os.path.join(new_epochs_ckpt_dir, "predicate_predict.txt"),
                                  "r", encoding='utf-8').readlines()
    predicate_score_value_file = open(os.path.join(new_epochs_ckpt_dir, "predicate_score_value.txt"),
                                      "r", encoding='utf-8').readlines()
    output_text_file_write = open(os.path.join(out_file, "text_and_one_predicate.txt"), "w", encoding='utf-8')
    output_token_in_file_write = open(os.path.join(out_file, "token_in_and_one_predicate.txt"), "w", encoding='utf-8')
    output_token_in_not_UNK_file_write = open(os.path.join(out_file, "token_in_not_UNK_and_one_predicate.txt"), "w", encoding='utf-8')
    count_line = 0
    count_empty_line = 0
    count_temporary_one_predicate_line = 0
    for text, token_in, token_in_not_UNK, predicate_predict, predicate_score_value in zip(text_file, token_in_file, token_in_not_UNK_file,
                                                                        predicate_predict_file, predicate_score_value_file):
        count_line += 1
        predicate_list = predicate_predict.replace("\n", "").split(" ")
        if predicate_predict == "\n":
            count_empty_line += 1
            predicate_list = replace_empty_infer_predicate_to_three_possible_values(predicate_score_value)
        for predicate in predicate_list:
            count_temporary_one_predicate_line += 1
            output_text_file_write.write(text.replace("\n", "") + "\t" + predicate + "\n")
            output_token_in_file_write.write(token_in.replace("\n", "") + "\t" + predicate + "\n")
            output_token_in_not_UNK_file_write.write(token_in_not_UNK.replace("\n", "") + "\t" + predicate + "\n")
    print("empty_line: {}, line: {}, empty percentage: {:.2f}%".format(count_empty_line, count_line, (count_empty_line/count_line) *100))
    print("temporary_one_predicate_line: ", count_temporary_one_predicate_line)
    print("输入文件行数：", count_line)
    print("转换成一个text 对应一个 predicate 之后行数变为：", count_temporary_one_predicate_line)


if __name__=="__main__":
    predicate_classifiction_input_file_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../predicate_classifiction/classification_data/test"))
    #predicate_classifiction_infer_file_dir = "output/predicate_infer_out/epochs6/ckpt23000"
    predicate_classifiction_infer_file_dir = None #None表示使用最新模型输出
    out_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../subject_object_labeling/sequence_labeling_data/test"))
    prepare_data_for_subject_object_labeling_infer(predicate_classifiction_input_file_dir,
                                                   predicate_classifiction_infer_file_dir, out_file)
