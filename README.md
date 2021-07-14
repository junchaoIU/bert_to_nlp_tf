# NLP 项目 的  tensorflow 实现

## 一、介绍

该项目主要 使用 tensorflow 框架 复现 NLP 基础任务上的应用。

## 二、项目架构

### 2.1 [文本分类任务](textClassifier/)

#### 2.1.1 [基于深度学习的分类方法](textClassifier/DL/)

- 目录文件：
  - train_model.py          模型训练文件
  - tf_model.py             模型预测
  - dl_model/               工具目录
    - dl_model/             模型
      - Config.py           配置类 【模型选择】
      - word2vec.py         词向量训练 模块
      - tools/              工具包
        - metrics.py        评测
        - logginger.py      日志
        - utils.py          数据处理模块 【数据预处理类Dataset、输出batch数据集 nextBatch】
        - utils_fasttext.py fasttext 数据处理类
      - layers/             模型网络 层 实现
        - textcnn  
        - textrnn
        - textrcnn
        - BiLSTMAttention
        - AdversarialLSTM
        - Transformer    

- 训练

```s
    python train_model.py
```

- 预测

```s
    python tf_model.py
```

#### 2.1.2 [基于 Bert 的分类方法](textClassifier/bert_task/)

- 目录文件：
  - bert/                   bert 项目
  - config/                 配置 文件
  - data/                   数据文件
  - trainer.py              训练 方法
  - predict_get_tensor.py   预测 基于 get_tensor_by_name
  - model.py                BertClassifier  类
  - data_helper.py          训练数据 处理类
  - metrics.py              评测

- 训练

1. 编写 config 目录下得 json 文件
2. 运行
```s
    python trainer.py --config_path=config/ccks_config.json
```

- 预测

```s
    python predict_get_tensor.py --config_path=config/ccks_config.json
```

### 2.2 [命名实体识别任务](NER/)

#### 2.2.1 [基于 Bert 的命名实体识别方法](textClassifier/ner_bert/)

- 目录文件：
  - bert/                   bert 项目
  - config/                 配置 文件
  - data/                   数据文件
  - trainer.py              训练 方法
  - predict_get_tensor.py   预测 基于 get_tensor_by_name
  - model.py                BertNer  类
  - bilstm_crf.py           BiLSTM-CRF 类
  - data_helper.py          训练数据 处理类
  - metrics.py              评测

- 训练

1. 编写 config 目录下得 json 文件
2. 运行
```s
    python trainer.py --config_path=config/ccks_config.json
```

- 预测

```s
    python predict_get_tensor.py --config_path=config/ccks_config.json
```

### 2.3 [文本相似度任务](Sim/)

#### 2.3.1 [基于 Bert 的文本匹配方法](textClassifier/sim_bert/)

- 目录文件：
  - bert/                   bert 项目
  - config/                 配置 文件
  - data/                   数据文件
  - trainer.py              训练 方法
  - predict_get_tensor.py   预测 基于 get_tensor_by_name
  - model.py                BertSentencePair  类
  - data_helper.py          训练数据 处理类
  - metrics.py              评测

- 训练

1. 编写 config 目录下得 json 文件
2. 运行
```s
    python trainer.py --config_path=config/ccks_config.json
```

- 预测

```s
    python predict_get_tensor.py --config_path=config/ccks_config.json
```

### 2.4 [阅读理解](Sim/)

#### 2.3.1 [基于 Bert 的文本匹配方法](textClassifier/machine_reading/)

- 目录文件：
  - bert/                   bert 项目
  - config/                 配置 文件
  - data/                   数据文件
  - trainer.py              训练 方法
  - predict.py              预测
  - model.py                BertSentencePair  类
  - data_helper.py          训练数据 处理类
  - metrics.py              评测

- 训练

1. 编写 config 目录下得 json 文件
2. 运行
```s
    python trainer.py --config_path=config/cmrc_config_linux.json
```

- 预测

```s
    python predict_get_tensor.py --config_path=config/cmrc_config_linux.json
```

## 三、软件 or python 包 版本

1.  python==3.6
2.  tensorflow <= 1.15.0
3.  jieba
4.  pandas
5.  numpy
6.  sklearn
7.  gensim
8.  fasttext

