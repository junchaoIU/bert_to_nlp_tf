#encoding:utf-8
import os
import pandas as pd
import json
import tensorflow as tf
import numpy as np
from dl_model.tools.utils import nextBatchTest
from dl_model.tools.tools import timer
import jieba

class TF_model(object):
    def __init__(self, config):
        self.model_path = config.savedCkptModelPath
        self.word2idxSource = config.word2idxSource
        self.label2idxSource = config.label2idxSource
        self.sequenceLength = config.sequenceLength
        self.batchSize = config.batchSize
        self.config = config
        jieba.load_userdict(f"{self.config.data_path}dict.txt")
        self.load_model()
        
    def load_model(self):
        # 注：下面两个词典要保证和当前加载的模型对应的词典是一致的
        with open(self.word2idxSource, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)

        with open(self.label2idxSource, "r", encoding="utf-8") as f:
            self.label2idx = json.load(f)
        self.idx2label = {value: key for key, value in self.label2idx.items()}
        
        #clean defualt graph
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                print(f"self.model_path:{self.model_path}")
                checkpoint_file = tf.train.latest_checkpoint(self.model_path)
                print("checkpoint_file:"+checkpoint_file)
                saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)

                # 获得需要喂给模型的参数，输出的结果依赖的输入值
                self.inputX = graph.get_operation_by_name("inputX").outputs[0]
                self.dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]

                # 获得输出的结果
                self.predictions = graph.get_tensor_by_name("output/predictions:0")
                self.score = graph.get_tensor_by_name("output/score:0")
            
    def process(self,sentence):
        xIds = [self.word2idx.get(item, self.word2idx["UNK"]) for item in list(jieba.cut(sentence, cut_all=False))]
        if len(xIds) >= self.sequenceLength:
            xIds = xIds[:self.sequenceLength]
        else:
            xIds = xIds + [self.word2idx["PAD"]] * (self.sequenceLength - len(xIds))
        return [xIds]
    
    @timer
    def predict(self,sentence):
        input_features = self.process(sentence)
        
        pred_list,score = self.sess.run([self.predictions,self.score], feed_dict={self.inputX: input_features, self.dropoutKeepProb: 1.0})
        pred_index = pred_list[0]
        pred_score = score[0]
        
        return pred_index,pred_score,self.idx2label[pred_index]

if __name__ == "__main__":
    tfmodel = TF_model(config)
    query_list = ['请把刘德华加入友群','十一月23号中午九点钟提醒我上毛概课','明天的运动指数',"我要看战狼"]
    for query in query_list:
        print(tfmodel.predict(query))