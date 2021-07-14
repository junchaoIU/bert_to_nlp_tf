import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from bert import modeling
from bert import optimization

class BertMachineReading(object):
    def __init__(self, config, is_training=True, num_train_step=None, num_warmup_step=None):
        self.__bert_config_path = os.path.join(
            config["bert_model_path"], "bert_config.json"
        )

        self.__is_training = is_training
        self.__num_train_step = num_train_step
        self.__num_warmup_step = num_warmup_step

        self.__max_length = config["max_length"]
        self.__learning_rate = config["learning_rate"]

        self.input_ids = tf.placeholder(
            dtype=tf.int32, shape=[None, self.__max_length], name='input_ids'
        )
        self.input_masks = tf.placeholder(
            dtype=tf.int32, shape=[None, self.__max_length], name='input_mask'
        )
        self.segment_ids = tf.placeholder(
            dtype=tf.int32, shape=[None, self.__max_length], name='segment_ids'
        )
        self.start_position = tf.placeholder(
            dtype=tf.int32, shape=[None], name="start_position"
        )
        self.end_position = tf.placeholder(
            dtype=tf.int32, shape=[None], name="end_position"
        )

        self.built_model()
        self.init_saver()

    # 功能：模型构建
    def built_model(self):
        '''
            功能：模型构建
        '''
        # 1. 加载 bert 模型训练参数
        bert_config = modeling.BertConfig.from_json_file(self.__bert_config_path)
        # 2. 加载 bert 模型
        model = modeling.BertModel(
            config=bert_config,
            is_training=self.__is_training,
            input_ids=self.input_ids,
            input_mask=self.input_masks,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False
        )
        # 3. 获取 Bert 最后一层 输出
        final_hidden = model.get_sequence_output()
        # 4. 返回 final_hidden 尺寸
        final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
        seq_length = final_hidden_shape[1]
        hidden_size = final_hidden_shape[2]
        # 5. 输出层 定义
        with tf.name_scope("output"):
            # 全连接
            output_weights = tf.get_variable(
                "output_weights", [2, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02)
            )
            output_bias = tf.get_variable(
                "output_bias", [2], initializer=tf.zeros_initializer()
            )
            final_hidden_matrix = tf.reshape(
                final_hidden,[-1, hidden_size]
            )
            logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            # 矩阵维度 调整 为  [-1, seq_length, 2]
            logits = tf.reshape(logits, [-1, seq_length, 2])
            # 矩阵转置  将 第三维转为 第一维，第一维转为第二维，第二维转为第三维 [2, -1, seq_length]
            logits = tf.transpose(logits, [2, 0, 1])
            # 矩阵 按 按 第一维 拆分  ( [-1, seq_length], [-1, seq_length] )
            unstacked_logits = tf.unstack(logits, axis=0)
            # [batch_size, seq_length]
            start_logits, end_logits = (unstacked_logits[0], unstacked_logits[1])

            self.start_logits = start_logits
            self.end_logits = end_logits
        # 6. 损失函数定义
        if self.__is_training:
            with tf.name_scope("loss"):
                # 计算 start 指针 的 loss 
                start_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=start_logits,
                    labels=self.start_position
                )
                # 计算 end 指针 的 loss 
                end_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=end_logits,
                    labels=self.end_position
                )
                # 合并两个指针 的 loss
                losses = tf.concat([start_losses, end_losses], axis=0)
                self.loss = tf.reduce_mean(losses, name="loss")
            # 7. loss 优化
            with tf.name_scope('train_op'):
                self.train_op = optimization.create_optimizer(
                    self.loss, self.__learning_rate, self.__num_train_step, self.__num_warmup_step, use_tpu=False
                )

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())

    # 功能：训练模型
    def train(self, sess, batch):
        """
            功能：训练模型
            input：
                :param sess: tf的会话对象
                :param batch: batch数据
            return:
                loss
                start_logits     首指针概率
                end_logits       尾指针概率
        """
        feed_dict = {
            self.input_ids: batch["input_ids"],
            self.input_masks: batch["input_masks"],
            self.segment_ids: batch["segment_ids"],
            self.start_position: batch["start_position"],
            self.end_position: batch["end_position"]
        }

        # 训练模型
        _, loss, start_logits, end_logits = sess.run(
            [self.train_op, self.loss, self.start_logits, self.end_logits], 
            feed_dict=feed_dict
        )
        return loss, start_logits, end_logits

    # 功能：模型验证
    def eval(self, sess, batch):
        """
            功能：模型验证
            input：
                :param sess: tf中的会话对象
                :param batch: batch数据
            :return: 预测结果
                start_logits     首指针概率
                end_logits       尾指针概率
        """
        feed_dict = {
            self.input_ids: batch["input_ids"],
            self.input_masks: batch["input_masks"],
            self.segment_ids: batch["segment_ids"],
            self.start_position: batch["start_position"],
            self.end_position: batch["end_position"]
        }

        start_logits, end_logits = sess.run(
            [self.start_logits, self.end_logits], feed_dict=feed_dict
        )
        return start_logits, end_logits
    # 功能：模型预测
    def infer(self, sess, batch):
        """
            功能：模型预测
            input：
                :param sess: tf中的会话对象
                :param batch: batch数据
            :return: 预测结果
                start_logits     首指针概率
                end_logits       尾指针概率
        """
        feed_dict = {
            self.input_ids: batch["input_ids"],
            self.input_masks: batch["input_masks"],
            self.segment_ids: batch["segment_ids"]
        }

        start_logits, end_logits = sess.run(
            [self.start_logits, self.end_logits], feed_dict=feed_dict
        )

        return start_logits, end_logits
