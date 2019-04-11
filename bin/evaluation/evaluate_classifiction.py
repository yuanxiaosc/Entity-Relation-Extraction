import os

golden_file_path = None
predicate_predict_file_path = None

#获取标准答案文件路径
def get_golden_file_path(golden_file_path=None):
    if golden_file_path is None:
        golden_file_dir = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "../predicate_classifiction/classification_data")), "test")
        golden_file_path = os.path.join(golden_file_dir, "predicate_out.txt")
    return golden_file_path


# 获取最新模型预测数据路径
def get_latest_model_predict_data_path(predicate_predict_file_path=None):
    # 获取文件下最新文件路径
    def new_report(test_report):
        lists = os.listdir(test_report)  # 列出目录的下所有文件和文件夹保存到lists
        lists.sort(key=lambda fn: os.path.getmtime(test_report + "/" + fn))  # 按时间排序
        file_new = os.path.join(test_report, lists[-1])  # 获取最新的文件保存到file_new
        return file_new
    if predicate_predict_file_path is None:
        # 获取分类预测输出文件路径
        input_new_epochs = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output")), "predicate_infer_out")
        # 获取最新周期文件路径
        new_ckpt_dir = new_report(input_new_epochs)
        input_new_epochs_ckpt = os.path.join(input_new_epochs, new_ckpt_dir)
        # 获取最新周期下最新模型文件路径
        input_new_epochs_ckpt_dir = new_report(input_new_epochs_ckpt)
        # 获取最新预测文件的路径
        predicate_predict_file_path = os.path.join(input_new_epochs_ckpt_dir, "predicate_predict.txt")
    if not os.path.exists(new_ckpt_dir):
        raise ValueError("路径不存在！{}".format(new_ckpt_dir))
    return predicate_predict_file_path


predicate_predict_file_path = get_latest_model_predict_data_path()
print("predicate_predict_file_path:\t", predicate_predict_file_path)

golden_file_path = get_golden_file_path()
print("golden_file_path:\t", golden_file_path)


golden_data = open(golden_file_path, "r", encoding='utf-8').readlines()
predict_data = open(predicate_predict_file_path, 'r', encoding='utf-8').readlines()
golden_data_list = [line.strip() for line in golden_data]
predict_data_list = [line.strip() for line in predict_data]
assert len(golden_data) == len(predict_data)

count_line = 0
correct_number = 0
subset_number = 0
superset_number = 0
for golden_str, predict_str in zip(golden_data_list, predict_data_list):
    count_line += 1
    golden_set = set(golden_str.split(" "))
    predict_set = set(predict_str.split(" "))
    if golden_set == predict_set:
        correct_number += 1
    elif predict_set.issubset(golden_set):
        subset_number += 1
    elif predict_set.issuperset(golden_set):
        superset_number +=1

print("correct_line: {}, line: {}, percentage: {:.4f}%".format(correct_number, count_line,
                                                                   (correct_number / count_line) * 100))
print("superset_line: {}, line: {}, percentage: {:.4f}%".format(superset_number, count_line,
                                                                   (superset_number / count_line) * 100))
print("subset_line: {}, line: {}, percentage: {:.4f}%".format(subset_number, count_line,
                                                                   (subset_number / count_line) * 100))
