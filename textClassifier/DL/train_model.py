import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import csv
import time
import datetime
import random
import pandas as pd
import tensorflow as tf
import json
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from dl_model.tools.metrics import *
from dl_model.tools.utils import Dataset,nextBatch
from dl_model.Config import Config
config = Config()

if config.layerType=="AdversarialLSTM":
    from collections import Counter
    class ADataset(Dataset):
        def __init__(self,config):  # 先继承，在重构
            Dataset.__init__(self,config)  #继承父类的构造方法，也可以写成：super(Chinese,self).__init__(name,age)
            self.indexFreqs = []  # 统计词空间中的词在出现在多少个review中
        # 生成词向量和词汇-索引映射字典，可以用全数据集
        def _genVocabulary(self, reviews, labels):
            """
            生成词向量和词汇-索引映射字典，可以用全数据集
            """
            allWords = [word for review in reviews for word in review]
            if self.isCleanStopWord:
                # 去掉停用词
                subWords = [word for word in allWords if word not in self.stopWordDict]
                wordCount = Counter(subWords)  # 统计词频
            else:
                wordCount = Counter(allWords)  # 统计词频
            sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
            # 去除低频词
            words = [item[0] for item in sortWordCount if item[1] >= 5]
            vocab, wordEmbedding = self._getWordEmbedding(words)
            self.wordEmbedding = wordEmbedding
            word2idx = dict(zip(vocab, list(range(len(vocab)))))
            uniqueLabel = list(set(labels))
            label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
            # 得到逆词频
            self._getWordIndexFreq(vocab, reviews, word2idx)
            self.labelList = list(range(len(uniqueLabel)))
            # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
            with open(self.word2idxSource, "w", encoding="utf-8") as f:
                json.dump(word2idx, f)
            with open(self.label2idxSource, "w", encoding="utf-8") as f:
                json.dump(label2idx, f)
            return word2idx, label2idx 
        def _getWordIndexFreq(self, vocab, reviews, word2idx):
            """
            统计词汇空间中各个词出现在多少个文本中
            """
            reviewDicts = [dict(zip(review, range(len(review)))) for review in reviews]
            indexFreqs = [0] * len(vocab)
            for word in vocab:
                count = 0
                for review in reviewDicts:
                    if word in review:
                        count += 1
                indexFreqs[word2idx[word]] = count
            self.indexFreqs = indexFreqs
    data = ADataset(config)
    data.dataGen()
else:
    data = Dataset(config)
    data.dataGen()

# 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels
wordEmbedding = data.wordEmbedding
labelList = data.labelList

# 定义计算图
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth=True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率  
    sess = tf.Session(config=session_conf)    
    # 定义会话
    with sess.as_default():   
        if config.layerType == "textCNN":
            from dl_model.layers.textcnn import TextCNN
            layer = TextCNN(config, wordEmbedding)
        elif config.layerType == "textRNN":
            from dl_model.layers.textrnn import BiLSTM
            layer = BiLSTM(config, wordEmbedding)
        elif config.layerType == "BiLSTMAttention":
            from dl_model.layers.BiLSTMAttention import BiLSTMAttention
            layer = BiLSTMAttention(config, wordEmbedding)
        elif config.layerType == "textrcnn":
            from dl_model.layers.textrcnn import RCNN
            layer = RCNN(config, wordEmbedding)
        elif config.layerType == "AdversarialLSTM":
            from dl_model.layers.AdversarialLSTM import AdversarialLSTM
            indexFreqs = data.indexFreqs
            layer = AdversarialLSTM(config, wordEmbedding,indexFreqs)
        elif config.layerType == "Transformer":
            from dl_model.layers.Transformer import Transformer,fixedPositionEmbedding
            embeddedPosition = fixedPositionEmbedding(config.batchSize, config.sequenceLength) 
            layer = Transformer(config, wordEmbedding)
        
        
        globalStep = tf.Variable(0, trainable=False, name='globalStep')
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        gradients, variables = zip(*optimizer.compute_gradients(layer.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, config.training.clip)
        # 将梯度应用到变量下，生成训练器
        gradsAndVars = zip(gradients, variables)
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)
        
        
        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        
        outDir = os.path.abspath(os.path.join(os.path.curdir, config.summarysPath))
        print("Writing to {}\n".format(outDir))        
        lossSummary = tf.summary.scalar("loss", layer.loss)
        summaryOp = tf.summary.merge_all()        
        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)        
        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)       
        
        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)        
        # 保存模型的一种方式，保存为pb文件
        if os.path.exists(config.savedPbModelPath):
            import shutil
            shutil.rmtree(config.savedPbModelPath)
        builder = tf.saved_model.builder.SavedModelBuilder(config.savedPbModelPath)            
        sess.run(tf.global_variables_initializer())
        def trainStep(batchX, batchY):
            """
            训练函数
            """   
            if config.layerType=="Transformer": 
                feed_dict = {
                  layer.inputX: batchX,
                  layer.inputY: batchY,
                  layer.dropoutKeepProb: config.dropoutKeepProb,
                  layer.embeddedPosition: embeddedPosition
                }
            else:
                feed_dict = {
                  layer.inputX: batchX,
                  layer.inputY: batchY,
                  layer.dropoutKeepProb: config.dropoutKeepProb,
                }
            _, summary, step, loss, predictions = sess.run(
                [trainOp, summaryOp, globalStep, layer.loss, layer.predictions],
                feed_dict)
            timeStr = datetime.datetime.now().isoformat()            
            trainSummaryWriter.add_summary(summary, step)            
            return loss # , acc, prec, recall, f_beta

        def devStep(batchX, batchY):
            """
            验证函数
            """
            if config.layerType=="Transformer": 
                feed_dict = {
                  layer.inputX: batchX,
                  layer.inputY: batchY,
                  layer.dropoutKeepProb: 1.0,
                  layer.embeddedPosition: embeddedPosition
                }
            else:
                feed_dict = {
                  layer.inputX: batchX,
                  layer.inputY: batchY,
                  layer.dropoutKeepProb: 1.0
                }
            summary, step, loss, predictions = sess.run(
                [summaryOp, globalStep, layer.loss, layer.predictions],
                feed_dict)
            
            if config.numClasses == 1:            
                acc, precision, recall, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)
            elif config.numClasses > 1:
                acc, precision, recall, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY, labels=labelList)            
            evalSummaryWriter.add_summary(summary, step)
            return loss, acc, precision, recall, f_beta
        bestF1Score = 0     # 验证集的最优 F1-score
        lastImproved = 0  # record global_step at best_val_accuracy
        requireImprovement = 1000  # break training if not having improvement over 1000 iter
        flag=False
        for i in range(config.training.epoches):
            # 训练模型
            print("start training model")
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                loss = trainStep(batchTrain[0], batchTrain[1])
                currentStep = tf.train.global_step(sess, globalStep) 
                print("train: step: {}, loss: {}".format(
                    currentStep, loss))
                if currentStep % config.training.evaluateEvery == 0:
                    print("\nEvaluation:")
                    losses = []
                    accs = []
                    f_betas = []
                    precisions = []
                    recalls = []
                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc, precision, recall, f_beta = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        f_betas.append(f_beta)
                        precisions.append(precision)
                        recalls.append(recall)
                           
                    improvedStr = ''
                    if bestF1Score<f_beta:
                        # 保存模型的另一种方法，保存checkpoint文件
                        bestF1Score=f_beta
                        improvedStr = '*'
                        lastImproved=currentStep
                        path = saver.save(sess, config.savedCkptModelPath, global_step=currentStep)
                        print("Saved model checkpoint to {}\n".format(path))
                    
                    timeStr = datetime.datetime.now().isoformat()
                    print("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}sec/batch {}".format(timeStr, currentStep, mean(losses), 
                                                                                                       mean(accs), mean(precisions),
                                                                                                        mean(recalls), mean(f_betas),improvedStr))
                    
                    if currentStep - lastImproved > requireImprovement:
                        print("No optimization over 1000 steps, stop training")
                        flag = True
                        break
            if flag:
                break
            config.training.learningRate *= config.training.lrDecay
                    
        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(layer.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(layer.dropoutKeepProb)}

        outputs = {"predictions": tf.saved_model.utils.build_tensor_info(layer.predictions)}

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                            signature_def_map={"predict": prediction_signature}, legacy_init_op=legacy_init_op)

        builder.save()