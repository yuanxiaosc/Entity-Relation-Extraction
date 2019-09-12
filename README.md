# Entity-Relation-Extraction
Entity and Relation Extraction Based on TensorFlow. 基于 TensorFlow 的实体及关系抽取，2019语言与智能技术竞赛信息抽取（实体与关系抽取）任务解决方案。

如果你对信息抽取论文研究感兴趣，可以查看我的博客 [望江人工智库 信息抽取](https://yuanxiaosc.github.io/categories/%E8%AE%BA%E6%96%87/%E4%BF%A1%E6%81%AF%E6%8A%BD%E5%8F%96/)。

## Abstract
该代码以管道式的方式处理实体及关系抽取任务，首先使用一个多标签分类模型判断句子的关系种类，然后把句子和可能的关系种类输入序列标注模型中，序列标注模型标注出句子中的实体，最终结合预测的关系和实体输出实体-关系列表：（实体1，关系，实体2）。

The code deals with entity and relationship extraction tasks in a pipeline way. First, a multi-label classification model is used to judge the relationship types of sentences. Then, the sentence and possible relationship types are input into the sequence labeling model. The sequence labeling model labels the entities in sentences, and finally combines the predicted relationship with the entity output entity-relationship list: (entity 1, relationship, entity 2).

整个实体关系抽取代码的具体细节和运行过程可以阅读 [bert实践:关系抽取解读](https://blog.csdn.net/weixin_42001089/article/details/97657149)，如果还有疑问或者想法欢迎提Issues :smile:

## [2019语言与智能技术竞赛](http://lic2019.ccf.org.cn/kg)

more info: 
1. [2019语言与智能技术竞赛](http://lic2019.ccf.org.cn) 
2. 比赛对应的论坛[语言与智能高峰论坛](http://tcci.ccf.org.cn/summit/2019/dl.php)
3. 比赛对应的会议 [NLPCC 2019](http://tcci.ccf.org.cn/conference/2019/cfpsw.php)

### 竞赛任务
给定schema约束集合及句子sent，其中schema定义了关系P以及其对应的主体S和客体O的类别，例如（S_TYPE:人物，P:妻子，O_TYPE:人物）、（S_TYPE:公司，P:创始人，O_TYPE:人物）等。 任务要求参评系统自动地对句子进行分析，输出句子中所有满足schema约束的SPO三元组知识Triples=[(S1, P1, O1), (S2, P2, O2)…]。
输入/输出:
(1) 输入:schema约束集合及句子sent
(2) 输出:句子sent中包含的符合给定schema约束的三元组知识Triples

**例子**
输入句子： ```"text": "《古世》是连载于云中书城的网络小说，作者是未弱"```

输出三元组： ```"spo_list": [{"predicate": "作者", "object_type": "人物", "subject_type": "图书作品", "object": "未弱", "subject": "古世"}, {"predicate": "连载网站", "object_type": "网站", "subject_type": "网络小说", "object": "云中书城", "subject": "古世"}]}```

### 数据简介
本次竞赛使用的SKE数据集是业界规模最大的基于schema的中文信息抽取数据集，其包含超过43万三元组数据、21万中文句子及50个已定义好的schema，表1中展示了SKE数据集中包含的50个schema及对应的例子。数据集中的句子来自百度百科和百度信息流文本。数据集划分为17万训练集，2万验证集和2万测试集。其中训练集和验证集用于训练，可供自由下载，测试集分为两个，测试集1供参赛者在平台上自主验证，测试集2在比赛结束前一周发布，不能在平台上自主验证，并将作为最终的评测排名。

## Getting Started
### Environment Requirements
+ python 3.6+
+ Tensorflow 1.12.0+

### Step 1: Environmental preparation
+ Install Tensorflow 
+ Dowload [bert-base, chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip), unzip file and put it in ```pretrained_model``` floader.

### Step 2: Download the training data, dev data and schema files
Please download the training data, development data and schema files from [the competition website](http://lic2019.ccf.org.cn/kg), then unzip files and put them in ```./raw_data/``` folder.
```
cd data
unzip train_data.json.zip 
unzip dev_data.json.zip
...
```

Official Data Download Address [baidu](http://ai.baidu.com/broad/download)

There is no longer a raw data download, if you have any questions, you can contact my mailbox wangzichaochaochao@gmail.com

**关系分类模型和实体序列标注模型可以同时训练，但是只能依次预测！**

## 训练阶段

### 准备关系分类数据
```
python bin/predicate_classifiction/predicate_data_manager.py
```

### 关系分类模型训练
```
python run_predicate_classification.py \
--task_name=SKE_2019 \
--do_train=true \
--do_eval=false \
--data_dir=bin/predicate_classifiction/classification_data \
--vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=6.0 \
--output_dir=./output/predicate_classification_model/epochs6/
```

### 准备序列标注数据
```
python bin/subject_object_labeling/sequence_labeling_data_manager.py
```

### 序列标注模型训练
```
python run_sequnce_labeling.py \
--task_name=SKE_2019 \
--do_train=true \
--do_eval=false \
--data_dir=bin/subject_object_labeling/sequence_labeling_data \
--vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=9.0 \
--output_dir=./output/sequnce_labeling_model/epochs9/
```

## 预测阶段

### 关系分类模型预测
```
python run_predicate_classification.py \
  --task_name=SKE_2019 \
  --do_predict=true \
  --data_dir=bin/predicate_classifiction/classification_data \
  --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/predicate_classification_model/epochs6/model.ckpt-27000 \
  --max_seq_length=128 \
  --output_dir=./output/predicate_infer_out/epochs6/ckpt27000
```

### 把关系分类模型预测结果转换成序列标注模型的预测输入
```
python bin/prepare_data_for_labeling_infer.py
```

### 序列标注模型预测
```
python run_sequnce_labeling.py \
  --task_name=SKE_2019 \
  --do_predict=true \
  --data_dir=bin/subject_object_labeling/sequence_labeling_data \
  --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/sequnce_labeling_model/epochs9/model.ckpt-22000 \
  --max_seq_length=128 \
  --output_dir=./output/sequnce_infer_out/epochs9/ckpt22000
```

### 生成实体-关系结果
```
python produce_submit_json_file.py
```

## 评估阶段
注意！官方提供的测试数据集 test1_data_postag.json 没有提供标签，所以只能提交给官方评测。
如果要自行评测模型效果：
```
predicate_data_manager.py set: Competition_Mode = False
```
然后运行：```bin/evaluation``` 中的评测文件


### 提交给官方评测的部分实验结果

|分类模型|序列标注模型|准确率|召回率|F1值|
|-|-|-|-|-|
|epochs6ckpt1000|epochs9ckpt4000|0.8549|0.7028|0.7714|
|epochs6ckpt13000|epochs9ckpt10000|0.8694|0.7188|0.7869|
|epochs6ckpt20000|epochs9ckpt17000|0.8651|0.738|0.7965|
|epochs6ckpt23000|epochs9ckpt20000|0.8714|0.7289|0.7938|

![](2019语言与智能技术竞赛信息抽取排行榜.png)

### 该任务的其它解决方案

+ [Baidu Official Baseline Model(Python2.7)](https://github.com/baidu/information-extraction) 
+ [Baseline Model(Python3)](https://github.com/yuanxiaosc/information-extraction)
+ [Multiple-Relations-Extraction-Only-Look-Once](https://github.com/yuanxiaosc/Multiple-Relations-Extraction-Only-Look-Once)
+ [Schema-based-Knowledge-Extraction](https://github.com/yuanxiaosc/Schema-based-Knowledge-Extraction)


### “信息抽取”任务冠军队伍报告

89.3% F1 在测试集，投入使用效果 87.1% F1，单模型，与本代码原理一致（见本资源Abstract部分）。

[Schema约束的知识抽取系统架构（“信息抽取”任务冠军队伍报告）](Schema约束的知识抽取系统架构（“信息抽取”任务冠军队伍报告）.pdf)
