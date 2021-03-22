import json
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
import time
import tensorflow as tf
from model import BertNer
from bert import tokenization
from metrics import get_chunk


class Predictor(object):
    def __init__(self, config_path):
        self.model = None
        with open(config_path, "r") as fr:
            self.config = json.load(fr)

        self.output_path = self.config["output_path"]
        self.vocab_path = os.path.join(self.config["bert_model_path"], "vocab.txt")
        self.label_to_index = self.load_vocab()
        self.word_vectors = None
        self.sequence_length = self.config["sequence_length"]

        # 创建模型
        self.create_model()
        # 加载计算图
        self.load_graph()

    def load_vocab(self):
        # 将词汇-索引映射表加载出来

        with open(os.path.join(self.output_path, "label_to_index.json"), "r") as f:
            label_to_index = json.load(f)

        return label_to_index

    def padding(self, input_id, input_mask, segment_id):
        """
        对序列进行补全
        :param input_id:
        :param input_mask:
        :param segment_id:
        :return:
        """

        if len(input_id) < self.sequence_length:
            pad_input_id = input_id + [0] * (self.sequence_length - len(input_id))
            pad_input_mask = input_mask + [0] * (self.sequence_length - len(input_mask))
            pad_segment_id = segment_id + [0] * (self.sequence_length - len(segment_id))
            sequence_len = len(input_id)
        else:
            pad_input_id = input_id[:self.sequence_length]
            pad_input_mask = input_mask[:self.sequence_length]
            pad_segment_id = segment_id[:self.sequence_length]
            sequence_len = self.sequence_length

        return pad_input_id, pad_input_mask, pad_segment_id, sequence_len

    def sentence_to_idx(self, text):
        """
        将分词后的句子转换成idx表示
        :return:
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_path, do_lower_case=True)

        tokens = []
        for token in text:
            token = tokenizer.tokenize(token)
            tokens.extend(token)

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_id = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_id)
        segment_id = [0] * len(input_id)

        input_id, input_mask, segment_id, sequence_len = self.padding(input_id, input_mask, segment_id)

        return [input_id], [input_mask], [segment_id], [sequence_len]

    def load_graph(self):
        """
        加载计算图
        :return:
        """
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.config["ckpt_model_path"])
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            print(f"ckpt.model_checkpoint_path:{ckpt.model_checkpoint_path}")
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(self.config["ckpt_model_path"]))

    def create_model(self):
        """
                根据config文件选择对应的模型，并初始化
                :return:
                """
        self.model = BertNer(config=self.config, is_training=False)

    def predict(self, text):
        """
        给定分词后的句子，预测其分类结果
        :param text:
        :return:
        """
        input_ids, input_masks, segment_ids, sequence_len = self.sentence_to_idx(text)

        prediction = self.model.infer(self.sess,
                                      dict(input_ids=input_ids,
                                           input_masks=input_masks,
                                           segment_ids=segment_ids,
                                           sequence_len=sequence_len)).tolist()
        chunks = get_chunk(prediction, self.label_to_index,  query)
        return chunks


if __name__ == "__main__":
    start = time.time()
    # 读取用户在命令行输入的信息
    config_path = "config/ccks_config.json"
    predictor = Predictor(config_path)
    end = time.time()
    print('加载模型所用时间:%s毫秒' % ((end - start)*1000))
    query_list = [
        "清朝词人纳兰性德是哪个民族的人？",
        "朱瞻基在位时的年号是什么？",
        "马尔科·巴萨是哪个国家的？",
        "国际商业机器公司有多少员工？"
    ]
    for query in query_list:
        start = time.time()
        chunks = predictor.predict(query)
        end = time.time()
        print(f"query:{query}->chunks:{chunks}->消耗时间：{(end - start)*1000}")