#encoding:utf-8
import tensorflow as tf
class ModelConfig(object):
    hiddenSizes = 128  # LSTM结构的神经元个数
    epsilon = 5
    
# 构建模型
class AdversarialLSTM(object):
    """
    Text CNN 用于文本分类
    """
    def __init__(self, config, wordEmbedding, indexFreqs):
        self.config = config
        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, self.config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None], name="inputY")
        
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        
        # 根据词的频率计算权重
        indexFreqs[0], indexFreqs[1] = 20000, 10000
        weights = tf.cast(tf.reshape(indexFreqs / tf.reduce_sum(indexFreqs), [1, len(indexFreqs)]), dtype=tf.float32)
        
        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用词频计算新的词嵌入矩阵
            normWordEmbedding = self._normalize(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), weights)
            
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(normWordEmbedding, self.inputX)
            
         # 计算二元交叉熵损失 
        with tf.name_scope("loss"):
            with tf.variable_scope("Bi-LSTM", reuse=None):
                self.logits = self._Bi_LSTMAttention(self.embeddedWords)
                
                if self.config.numClasses == 1:
                    self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name="predictions")
                    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(tf.reshape(self.inputY, [-1, 1]), 
                                                                                                    dtype=tf.float32))
                elif self.config.numClasses > 1:
                    self.maxScore= tf.nn.softmax(self.logits, name="score")
                    self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
                
                loss = tf.reduce_mean(losses)
        
        with tf.name_scope("perturLoss"):
            with tf.variable_scope("Bi-LSTM", reuse=True):
                perturWordEmbedding = self._addPerturbation(self.embeddedWords, loss)
                perturPredictions = self._Bi_LSTMAttention(perturWordEmbedding)
                perturLosses = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=perturPredictions, 
                    labels=tf.cast(tf.reshape(self.inputY, [-1, 1]), 
                    dtype=tf.float32))
                perturLoss = tf.reduce_mean(perturLosses)
        
        self.loss = loss + perturLoss
            
    def _Bi_LSTMAttention(self, embeddedWords):
        """
        Bi-LSTM + Attention 的模型结构
        """
        
        # 定义双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
           
            # 定义前向LSTM结构
            lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(
                num_units=self.config.model.hiddenSizes, state_is_tuple=True),
                output_keep_prob=self.dropoutKeepProb
            )
            # 定义反向LSTM结构
            lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(
                num_units=self.config.model.hiddenSizes, state_is_tuple=True),
                output_keep_prob=self.dropoutKeepProb
            )


            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
            # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
            # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
            outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(
                lstmFwCell, 
                lstmBwCell, 
                self.embeddedWords, 
                dtype=tf.float32,
                scope="bi-lstm"
            )

        
        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            output = self._attention(H)
            outputSize = self.config.model.hiddenSizes
        
        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, self.config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())
            
            outputB= tf.Variable(tf.constant(0.1, shape=[self.config.numClasses]), name="outputB")
            predictions = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")
            
        return predictions
    
    def _attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.config.model.hiddenSizes
        
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)
        
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.config.sequenceLength])
        
        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.config.sequenceLength, 1]))
        
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)
        
        return output
    
    def _normalize(self, wordEmbedding, weights):
        """
        对word embedding 结合权重做标准化处理
        """
        
        mean = tf.matmul(weights, wordEmbedding)
        print(mean)
        powWordEmbedding = tf.pow(wordEmbedding - mean, 2.)
        
        var = tf.matmul(weights, powWordEmbedding)
        print(var)
        stddev = tf.sqrt(1e-6 + var)
        
        return (wordEmbedding - mean) / stddev
    
    def _addPerturbation(self, embedded, loss):
        """
        添加波动到word embedding
        """
        grad, = tf.gradients(
            loss,
            embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)
        perturb = self._scaleL2(grad, self.config.model.epsilon)
        return embedded + perturb
    
    def _scaleL2(self, x, norm_length):
        # shape(x) = (batch, num_timesteps, d)
        # Divide x by max(abs(x)) for a numerically stable L2 norm.
        # 2norm(x) = a * 2norm(x/a)
        # Scale over the full sequence, dims (1, 2)
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keepdims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(
            tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keepdims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit