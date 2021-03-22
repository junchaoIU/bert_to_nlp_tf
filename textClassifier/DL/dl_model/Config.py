#encoding:utf-8
# 配置参数
class TrainingConfig(object):
    epoches = 100
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 1e-3
    lrDecay= 0.5          #learning rate decay
    clip= 5.0              #gradient clipping threshold
    l2RegLambda=0.05     #l2 regularization lambda

class Config():
    def __init__(self,data_type):
        ''''
            全局信息
        '''
        self.resource_path = "data/resource/"
        self.data_path = f"data/{data_type}/"
        self.output_path = f"data/{data_type}/"
        self.log_dir = f"{self.output_path}log/"

        '''
            词向量 训练
        '''
        self.file_name = f"{self.data_path}all.txt"
        self.key = "query"
        self.embeddingSize = 100
        self.window = 5
        self.embedding_dir = self.output_path
        self.tag = "word"
        self.train_cut = f"{self.output_path}trainCut.csv"
        self.embedingSource = '%s%s_word2vec_size%d_win%d'%(self.output_path,self.tag,self.embeddingSize,self.window) +'.txt'

        '''
            模型训练
        '''
        ## 模型参数设置
        self.layerType = "textCNN"
        '''
            linux：服务器 
            local：本地 batch
            single: 本地句子预测
        '''
        self.status = "linux" 
        self.sequenceLength = 32  # 取了所有序列长度的均值
        self.batchSize = 128
        self.textName = "query"
        self.labelName = "label"
        self.isCleanStopWord = False
        self.numClasses = 14  # 二分类设置为1，多分类设置为类别的数目
        self.rate = 0.7  # 训练集的比例
        self.dropoutKeepProb = 0.5 
        self.wordvecType=""
        self.training = TrainingConfig()
        self.wordFred = 1
        self.textStatus =  ''
        self.iter = 50
        self.training.evaluateEvery =100
        self.training.checkpointEvery =100
        self.textStatus =  ''
        if self.status=="linux":
            self.training.epoches =100
        else:
            self.training.epoches =10
            
        ## 模型 资源文件
        self.dataSource = f"{self.output_path}trainCut.csv"
        self.testFileSourceOutput = self.output_path+self.layerType+"_pred_score"+self.textStatus+".csv"
        self.stopWordSource = self.output_path+"stopword.txt"
        self.word2idxSource = self.output_path+"word2idx"+self.textStatus+".json"
        self.label2idxSource = self.output_path+"label2idx"+self.textStatus+".json"
        self.summarysPath = f"{self.output_path}summarys/{self.layerType}"
        self.savedPbModelPath = f"{self.output_path}model/{self.layerType}/savedModel"
        self.savedCkptModelPath = f"{self.output_path}model/{self.layerType}/model/"

        ## 模型专属配置加载 
        if self.layerType == "textCNN":
            from dl_model.layers.textcnn import ModelConfig
        elif self.layerType=="textRNN":
            from dl_model.layers.textrnn import ModelConfig
        elif self.layerType=="BiLSTMAttention":
            from dl_model.layers.BiLSTMAttention import ModelConfig
        elif self.layerType=="textrcnn":
            from dl_model.layers.textrcnn import ModelConfig
        elif self.layerType=="AdversarialLSTM":
            from dl_model.layers.AdversarialLSTM import ModelConfig
        elif self.layerType=="Transformer":
            from dl_model.layers.Transformer import ModelConfig
        self.model = ModelConfig()