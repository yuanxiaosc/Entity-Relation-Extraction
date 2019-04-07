# -*- coding: utf-8 -*-
########################################################
# Copyright (c) 2019, Baidu Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# imitations under the License.
########################################################
"""
This module to calculate precision, recall and f1-value
of the predicated results.
"""
import sys
import json
import os
import zipfile
import traceback
import argparse
import io
import configparser

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
SUCCESS = 0
FILE_ERROR = 1
ENCODING_ERROR = 2
JSON_ERROR = 3
SCHEMA_ERROR = 4
TEXT_ERROR = 5
CODE_INFO = ['success', 'file_reading_error', 'encoding_error', 'json_parse_error',
             'schema_error', 'input_text_not_in_dataset']


def del_bookname(entity_name):
    """delete the book name"""
    if entity_name.startswith(u'《') and entity_name.endswith(u'》'):
        entity_name = entity_name[1:-1]
    return entity_name


def load_predict_result(predict_filename):
    """Loads the file to be predicted"""
    predict_result = {}
    ret_code = SUCCESS
    try:
        predict_file_zip = zipfile.ZipFile(predict_filename)
    except:
        ret_code = FILE_ERROR
        return predict_result, ret_code
    for predict_file in predict_file_zip.namelist():
        for line in predict_file_zip.open(predict_file):
            try:
                line = line.strip()
            except:
                ret_code = ENCODING_ERROR
                return predict_result, ret_code
            try:
                json_info = json.loads(line)
            except:
                ret_code = JSON_ERROR
                return predict_result, ret_code
            if 'text' not in json_info or 'spo_list' not in json_info:
                ret_code = SCHEMA_ERROR
                return predict_result, ret_code
            sent = json_info['text']
            spo_set = set()
            for spo_item in json_info['spo_list']:
                if type(spo_item) is not dict or 'subject' not in spo_item \
                        or 'predicate' not in spo_item \
                        or 'object' not in spo_item or \
                        not isinstance(spo_item['subject'], str) or \
                        not isinstance(spo_item['object'], str):
                    ret_code = SCHEMA_ERROR
                    return predict_result, ret_code
                s = del_bookname(spo_item['subject'].lower())
                o = del_bookname(spo_item['object'].lower())
                spo_set.add((s, spo_item['predicate'], o))
            predict_result[sent] = spo_set
    return predict_result, ret_code


def load_test_dataset(golden_filename):
    """load golden file"""
    golden_dict = {}
    ret_code = SUCCESS
    with open(golden_filename, 'r', encoding='utf-8') as gf:
        for line in gf:
            try:
                line = line.strip()
            except:
                ret_code = ENCODING_ERROR
                return golden_dict, ret_code
            try:
                json_info = json.loads(line)
            except:
                ret_code = JSON_ERROR
                return golden_dict, ret_code
            try:
                sent = json_info['text']
                spo_list = json_info['spo_list']
            except:
                ret_code = SCHEMA_ERROR
                return golden_dict, ret_code

            spo_result = []
            for item in spo_list:
                o = del_bookname(item['object'].lower())
                s = del_bookname(item['subject'].lower())
                spo_result.append((s, item['predicate'], o))
            spo_result = set(spo_result)
            golden_dict[sent] = spo_result
    return golden_dict, ret_code


def load_dict(dict_filename):
    """load alias dict"""
    alias_dict = {}
    ret_code = SUCCESS
    if dict_filename == "":
        return alias_dict, ret_code
    try:
        with open(dict_filename, 'r', encoding='utf-8') as af:
            for line in af:
                line = line.strip()
                words = line.split('\t')
                alias_dict[words[0].lower()] = set()
                for alias_word in words[1:]:
                    alias_dict[words[0].lower()].add(alias_word.lower())
    except:
        ret_code = FILE_ERROR
    return alias_dict, ret_code


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


def calc_pr(predict_filename, alias_filename, location_filename, golden_filename):
    """calculate precision, recall, f1"""
    ret_info = {}
    # load location dict
    loc_dict, ret_code = load_dict(location_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        print('loc file is error')
        return ret_info

    # load alias dict
    alias_dict, ret_code = load_dict(alias_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        print('alias file is error')
        return ret_info
    # load test dataset
    golden_dict, ret_code = load_test_dataset(golden_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        print('golden file is error')
        return ret_info
    # load predict result
    predict_result, ret_code = load_predict_result(predict_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        print('predict file is error')
        return ret_info

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
    print(sys.stderr, 'correct spo num = ', correct_sum)
    print(sys.stderr, 'submitted spo num = ', predict_sum)
    print(sys.stderr, 'golden set spo num = ', recall_sum)
    precision = correct_sum / predict_sum if predict_sum > 0 else 0.0
    recall = correct_sum / recall_sum if recall_sum > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall > 0 else 0.0
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    ret_info['errorCode'] = SUCCESS
    ret_info['errorMsg'] = CODE_INFO[SUCCESS]
    ret_info['data'] = []
    ret_info['data'].append({'name': 'precision', 'value': precision})
    ret_info['data'].append({'name': 'recall', 'value': recall})
    ret_info['data'].append({'name': 'f1-score', 'value': f1})
    return ret_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden_file", type=str,
                        help="true spo results", required=True)
    parser.add_argument("--predict_file", type=str,
                        help="spo results predicted", required=True)
    parser.add_argument("--loc_file", type=str,
                        default='', help="location entities of various granularity")
    parser.add_argument("--alias_file", type=str,
                        default='', help="entities alias dictionary")
    args = parser.parse_args()
    golden_filename = args.golden_file
    predict_filename = args.predict_file
    location_filename = args.loc_file
    alias_filename = args.alias_file
    ret_info = calc_pr(predict_filename, alias_filename, location_filename, golden_filename)
    print(json.dumps(ret_info))
