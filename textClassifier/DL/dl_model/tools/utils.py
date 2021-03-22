#encoding:utf-8
import json
from collections import Counter
from math import sqrt
import gensim
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
# 数据预处理的类，生成训练集和测试集
class Dataset(object):
    def __init__(self, config):
        self.config = config
        self._dataSource = config.dataSource
        self.textName = config.textName
        self.labelName = config.labelName
        
        self.isCleanStopWord = config.isCleanStopWord
        self._stopWordSource = config.stopWordSource  
        
        self._embedingSource = config.embedingSource 
        self._embeddingSize = config.embeddingSize
        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        
        self.word2idxSource = config.word2idxSource 
        self.label2idxSource = config.label2idxSource
        self.wordFred =config.wordFred
        
        self._batchSize = config.batchSize
        self._rate = config.rate
        self._stopWordDict = {}
        self.trainReviews = []
        self.trainLabels = []
        self.evalReviews = []
        self.evalLabels = []
        self.wordEmbedding =None
        self.labelList = []
    
    # 从csv文件中读取数据集
    def _readData(self, filePath,isSample=True):
        """
        从csv文件中读取数据集
        """
        df = pd.read_csv(filePath,sep="\t",encoding="utf-8")
        if isSample:
            df = df.sample(frac=1)
        if self.config.numClasses == 1:
            labels = df[self.labelName].tolist()
        elif self.config.numClasses > 1:
            labels = df[self.labelName].tolist() 
        review = df[self.textName].tolist()
        del df
        # print(f"review:{review[0:1]}")
        reviews = [line.strip().split() for line in review]
        del review
        # print(f"reviews[0:1]:{reviews[0:1]}")
        return reviews, labels
    
    # 将标签转换成索引表示
    def _labelToIndex(self, labels, label2idx):
        """
        将标签转换成索引表示
        """
        labelIds = [label2idx[label] for label in labels]
        return labelIds
    
    # 将词转换成索引
    def _wordToIndex(self, reviews, word2idx):
        """
        将词转换成索引
        """
        reviewIds = [[word2idx.get(item, word2idx["UNK"]) for item in review] for review in reviews]
        return reviewIds
    
    # 生成训练集和验证集
    def _genTrainEvalData(self, x, y, word2idx, rate):
        """
        生成训练集和验证集
        """
        reviews = []
        for review in x:
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))

        # trainReviews = np.asarray(reviews, dtype="int64")
        # trainLabels = np.array(y, dtype="float32")
        # trainReviews, evalReviews, trainLabels, evalLabels = train_test_split(trainReviews, trainLabels, test_size=0.2, random_state=0)

        trainIndex = int(len(x) * rate)
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(y[:trainIndex], dtype="float32")
        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(y[trainIndex:], dtype="float32")
        return trainReviews, trainLabels, evalReviews, evalLabels 
    
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
        words = [item[0] for item in sortWordCount if item[1] >= self.wordFred]
        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding
        word2idx = dict(zip(vocab, list(range(len(vocab)))))
        uniqueLabel = list(set(labels))
        label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
        self.labelList = list(range(len(uniqueLabel)))
        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open(self.word2idxSource, "w", encoding="utf-8") as f:
            json.dump(word2idx, f)
        with open(self.label2idxSource, "w", encoding="utf-8") as f:
            json.dump(label2idx, f)
        return word2idx, label2idx
    
    # 按照我们的数据集中的单词取出预训练好的word2vec中的词向量
    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """
        wordVec = gensim.models.KeyedVectors.load_word2vec_format(self._embedingSource)
        vocab = []
        wordEmbedding = []
        # 添加 "pad" 和 "UNK", 
        vocab.append("PAD")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))
        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")
                
        return vocab, np.array(wordEmbedding)
    
    # 读取停用词
    def _readStopWord(self, stopWordPath):
        """
        读取停用词
        """
        with open(stopWordPath, "r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))
    
    # 初始化训练集和验证集  
    def dataGen(self):
        """
        初始化训练集和验证集
        """
        # # 初始化停用词
        if self.isCleanStopWord:
            self._readStopWord(self._stopWordSource)
        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)
        # 初始化词汇-索引映射表和词向量矩阵
        word2idx, label2idx = self._genVocabulary(reviews, labels)
        # 将标签和句子数值化
        labelIds = self._labelToIndex(labels, label2idx)
        reviewIds = self._wordToIndex(reviews, word2idx)
        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviewIds, labelIds, word2idx, self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels
        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

# 输出batch数据集
def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]
    numBatches = len(x) // batchSize
    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")
        yield batchX, batchY 

# 输出batch数据集
def nextBatchTest(x, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """
    x = np.asarray(x, dtype="int64")
    if len(x)%batchSize==0:
        numBatches = len(x) // batchSize
    else:
        numBatches = len(x) // batchSize + 1
    print(f"numBatches:{numBatches}")
    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        yield batchX 

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)