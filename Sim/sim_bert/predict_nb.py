import json
import os
import sys
import tensorflow as tf
from .model_nb import BertSentencePair
from .bert import tokenization


class SimPredictor(object):
    def __init__(self, config_path,config_file):
        self.model = None
        self.config_path = config_path
        with open(config_file, "r") as fr:
            self.config = json.load(fr)

        self.output_path = f'{self.config_path}{self.config["output_path"]}'
        self.vocab_path = os.path.join(self.config["bert_model_path"], "vocab.txt")
        self.label_to_index = self.load_vocab()
        self.index_to_label = {value: key for key, value in self.label_to_index.items()}
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

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_len):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_len:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

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
        else:
            pad_input_id = input_id[:self.sequence_length]
            pad_input_mask = input_mask[:self.sequence_length]
            pad_segment_id = segment_id[:self.sequence_length]

        return pad_input_id, pad_input_mask, pad_segment_id

    def sentence_to_idx(self, text_a, text_b):
        """
        将分词后的句子转换成idx表示
        :return:
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_path, do_lower_case=True)

        text_a = tokenization.convert_to_unicode(text_a)
        text_b = tokenization.convert_to_unicode(text_b)
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)

        # 判断两条序列组合在一起长度是否超过最大长度
        self._truncate_seq_pair(tokens_a, tokens_b, self.sequence_length - 3)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_id)
        segment_id = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

        input_id, input_mask, segment_id = self.padding(input_id, input_mask, segment_id)

        return [input_id], [input_mask], [segment_id]

    def load_graph(self):
        """
        加载计算图
        :return:
        """
        tf.reset_default_graph()
        graph2 = tf.Graph()
        self.sess = tf.Session(graph=graph2)
        with graph2.as_default():
            self.sess.run(tf.global_variables_initializer())
            checkpoint_file = tf.train.latest_checkpoint(f"{self.config_path}{self.config['ckpt_model_path']}")
            print("checkpoint_file:"+checkpoint_file)
            saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(self.sess, checkpoint_file)

            #输入op
            self.input_ids = graph2.get_tensor_by_name('input_ids:0')
            self.input_masks = graph2.get_tensor_by_name('input_mask:0')
            self.segment_ids = graph2.get_tensor_by_name('segment_ids:0')

            # 获得输出的结果
            self.predictions = graph2.get_tensor_by_name("output/predictions:0")

    def create_model(self):
        """
                根据config文件选择对应的模型，并初始化
                :return:
                """
        self.model = BertSentencePair(config=self.config, is_training=False)

    def predict(self, text_a, text_b):
        """
        给定分词后的句子，预测其分类结果
        :param text_a:
        :param text_b:
        :return:
        """
        input_ids, input_masks, segment_ids = self.sentence_to_idx(text_a, text_b)

        pred_list = self.sess.run(
            [self.predictions], 
            feed_dict={
                self.input_ids:input_ids,
                self.input_masks:input_masks,
                self.segment_ids:segment_ids
            }
        )[0]

        print(f"pred_list:{pred_list.tolist()}")
        prediction = pred_list.tolist()[0]
        label = self.index_to_label[prediction]

        return label

if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    config_path = ""
    config_file = f"{config_path}config/ccks_config.json"
    sim_predicter = Sim_Predictor(config_path,config_file)
    while True:
        sentence1 = input("输入句子1：")
        sentence2 = input("输入句子2：")
        label = sim_predicter.predict(sentence1,sentence2)
        print(f"label:{label}")
