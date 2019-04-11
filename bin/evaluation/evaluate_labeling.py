import sys
import json
import os

def del_bookname(entity_name):
    """delete the book name"""
    if entity_name.startswith(u'《') and entity_name.endswith(u'》'):
        entity_name = entity_name[1:-1]
    return entity_name




def load_result(predict_filename):
    result_dict = {}
    with open(predict_filename) as gf:
        for line in gf:
            json_info = json.loads(line)
            sent = json_info['text']
            spo_list = json_info['spo_list']
            spo_result = []
            for item in spo_list:
                o = del_bookname(item['object'].lower())
                s = del_bookname(item['subject'].lower())
                spo_result.append((s, item['predicate'], o))
            spo_result = set(spo_result)
            result_dict[sent] = spo_result
    return result_dict


def load_dict(dict_filename):
    """load alias dict"""
    alias_dict = {}
    with open(dict_filename) as af:
        for line in af:
            line = line.strip()
            words = line.split('\t')
            alias_dict[words[0].lower()] = set()
            for alias_word in words[1:]:
                alias_dict[words[0].lower()].add(alias_word.lower())
    return alias_dict


def is_spo_correct(spo, golden_spo_set, alias_dict, loc_dict):
    """if the spo is correct"""
    if spo in golden_spo_set:
        return True
    (s, p, o) = spo
    # alias dictionary
    s_alias_set = alias_dict.get(s, set())
    s_alias_set.add(s)
    o_alias_set = alias_dict.get(o, set())
    o_alias_set.add(o)
    for s_a in s_alias_set:
        for o_a in o_alias_set:
            if (s_a, p, o_a) in golden_spo_set:
                return True
    for golden_spo in golden_spo_set:
        (golden_s, golden_p, golden_o) = golden_spo
        golden_o_set = loc_dict.get(golden_o, set())
        for g_o in golden_o_set:
            if s == golden_s and p == golden_p and o == g_o:
                return True
    return False


def calc_pr(predict_filename, golden_filename):
    """calculate precision, recall, f1"""
    alias_dict, loc_dict = dict(), dict()
    ret_info = {}
    # load test dataset
    golden_dict= load_result(golden_filename)
    # load predict result
    predict_result = load_result(predict_filename)
    # evaluation
    correct_sum, predict_sum, recall_sum = 0.0, 0.0, 0.0
    for sent in golden_dict:
        golden_spo_set = golden_dict[sent]
        predict_spo_set = predict_result.get(sent, set())

        recall_sum += len(golden_spo_set)
        predict_sum += len(predict_spo_set)
        for spo in predict_spo_set:
            if is_spo_correct(spo, golden_spo_set, alias_dict, loc_dict):
                correct_sum += 1
    print('correct spo num = ', correct_sum)
    print('submitted spo num = ', predict_sum)
    print('golden set spo num = ', recall_sum)
    precision = correct_sum / predict_sum if predict_sum > 0 else 0.0
    recall = correct_sum / recall_sum if recall_sum > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall > 0 else 0.0
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    ret_info['data'] = []
    ret_info['data'].append({'name': 'precision', 'value': precision})
    ret_info['data'].append({'name': 'recall', 'value': recall})
    ret_info['data'].append({'name': 'f1-score', 'value': f1})
    return ret_info

if __name__ == '__main__':
    golden_filename = None
    predict_filename = None
    golden_filename = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../raw_data")), "dev_data.json")
    predict_filename = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output/final_text_spo_list_result")),
        "keep_empty_spo_list_subject_predicate_object_predict_output.json")
    ret_info = calc_pr(predict_filename, golden_filename)
    print(json.dumps(ret_info))