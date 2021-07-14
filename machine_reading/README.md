# 【关于 基于 Bert 阅读理解】那些你不知道的事


## 一、运行环境

- python == 3.6+
- tensorflow == 1.15.0

## 二、运行方式

1. 训练

```s
    !python trainer.py --config_path=config/cmrc_config.json 
    >>>
    WARNING:tensorflow:
    The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
    * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
    * https://github.com/tensorflow/addons
    * https://github.com/tensorflow/io (for I/O related ops)
    If you depend on functionality not listed there, please file an issue.

    2021-07-06 19:31:07.898776: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2021-07-06 19:31:07.899047: E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: UNKNOWN ERROR (303)
    2021-07-06 19:31:07.899075: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (ubuntu): /proc/driver/nvidia/version does not exist
    2021-07-06 19:31:07.899506: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
    2021-07-06 19:31:07.945702: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2100000000 Hz
    2021-07-06 19:31:07.960885: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5644e77a5a90 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
    2021-07-06 19:31:07.960963: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
    WARNING:tensorflow:From trainer.py:62: The name tf.train.init_from_checkpoint is deprecated. Please use tf.compat.v1.train.init_from_checkpoint instead.

    WARNING:tensorflow:From trainer.py:64: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.

    read finished
    index transform finished
    DEV_101_QUERY_3
    灭司马家族
    []
    司马家族，
    DEV_110_QUERY_2
    新界西贡区清水湾半岛以南的大庙湾地堂咀
    []
    西贡区清水湾半岛以南的大庙湾地堂咀，邻
    DEV_110_QUERY_3
    庙宇三面环山，而且那些山区都是郊野公园范围
    []
    三面环山，而且那些山区都是郊野公园范围，所
    read finished
    index transform finished
    train data size: 17095
    eval data size: 5536
    init bert model params
    init bert model params done
    ----- Epoch 1/10 -----
    train: time: 2021-07-06 19:31:30, step: 0, loss: 6.330463886260986
    train: time: 2021-07-06 19:31:36, step: 1, loss: 6.128863334655762
    train: time: 2021-07-06 19:31:45, step: 2, loss: 6.301714897155762
    train: time: 2021-07-06 19:31:52, step: 3, loss: 6.212819576263428
    train: time: 2021-07-06 19:31:57, step: 4, loss: 6.159421920776367
    train: time: 2021-07-06 19:32:03, step: 5, loss: 6.186403274536133
    train: time: 2021-07-06 19:32:09, step: 6, loss: 6.309201240539551
    train: time: 2021-07-06 19:32:15, step: 7, loss: 6.254149913787842
    train: time: 2021-07-06 19:32:21, step: 8, loss: 6.286585807800293
    train: time: 2021-07-06 19:32:27, step: 9, loss: 6.383264541625977
    train: time: 2021-07-06 19:32:33, step: 10, loss: 6.13147497177124
    ...
```

2. 预测


## 三、config 配置文件解读

> 以 cmrc_config.json为例

* model_name：模型名称
* epochs：迭代epoch的数量
* checkpoint_every：间隔多少步保存一次模型
* eval_every：间隔多少步验证一次模型
* learning_rate：学习速率，推荐2e-5， 5e-5， 1e-4
* max_length：输入到模型中的最大长度，建议设置为512
* doc_stride：对于context长度较长的时候会分成多个doc，采用滑动窗口的形式分doc，这个是滑动窗口的大小，建议设为128
* query_length：输入的问题的最大长度
* max_answer_length：生成的回答的最大长度
* n_best_size：获取分数最高的前n个
* batch_size：单GPU时不要超过32
* num_classes：文本分类的类别数量
* warmup_rate：训练时的预热比例，建议0.05， 0.1
* output_path：输出文件夹，用来存储label_to_index等文件
* output_predictions_path：训练时在验证集上预测的最佳结果保存路径
* output_nbest_path：训练时在验证集上预测的n个最佳结果的保存路径
* bert_model_path：预训练模型文件夹路径
* train_data：训练数据路径
* eval_data：验证数据路径
* ckpt_model_path：checkpoint模型文件保存路径

## 四、原始数据介绍

> 以 cmrc_config.json为例

- 训练数据介绍

```json
{
  "version": "v1.0", 
  "data": [
    {
      "paragraphs": [
        {
          "id": "TRAIN_186", 
          "context": "范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；... 范廷颂于2009年2月22日清晨在河内离世，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。", 
          "qas": [
            {
              "question": "范廷颂是什么时候被任为主教的？", 
              "id": "TRAIN_186_QUERY_0", 
              "answers": [
                {
                  "text": "1963年", 
                  "answer_start": 30
                }
              ]
            }, ...
          ]
        }
      ], 
      "id": "TRAIN_186", 
      "title": "范廷颂"
    }, ...
  ]
}
```

> 这里主要用到 数据中 的 "data" 模块

- paragraphs：段落
  - id：段落 id
  - context：段落内容
  - qas：问题列表
    - question：问题
    - id：问题 id
    - answers：问题答案
      - text：答案内容
      - answer_start：答案在 段落中的起始位置
- id：文章 id
- title：文章标题

## 五、代码实现分析

### 5.1  TrainData 类 【data_helper.py】

#### 5.1.1 整体结构

```python
class TrainData(object):
    def __init__(self, config):
        self.__vocab_path = os.path.join(config["bert_model_path"], "vocab.txt")
        self.__output_path = config["output_path"]
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)

        self.__query_length = config["query_length"]
        self.__doc_stride = config["doc_stride"]
        self.__max_length = config["max_length"]
        self.__batch_size = config["batch_size"]

    def _improve_answer_span(
      self, doc_tokens, input_start, input_end, tokenizer,orig_answer_text
    ):
        """
            功能：返回与标注答案更好匹配的标记化答案范围。
            input：
                doc_tokens
                input_start
                input_end
                tokenizer
                orig_answer_text
            :return:
                (input_start, input_end)
        """

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """
            功能：Check if this is the 'max context' doc span for the token.
            input：
              doc_spans
              cur_span_index
              position
            return：
              cur_span_index == best_span_index
        """

    # 功能：将原文分割成列表返回，主要是确保一连串的数字，或者英文单词作为一个单独的token存在
    def _split_char(self, context):
        """
            功能：将原文分割成列表返回，主要是确保一连串的数字，或者英文单词作为一个单独的token存在
            input：
                :param context:
            :return:
                new_context
            eg：
              content：1950年代
              new_context：["1958","年","代"]
        """

    # 功能：加载数据
    def read_data(self, file_path, is_training):
        """
            功能：加载数据
            input：
                file_path:        String       数据文件
                is_training       boolean      是否训练
            :return:
                examples
        """

    # 功能：将输入转化为索引表示
    def trans_to_features(self, examples, is_training):
        """
            功能：将输入转化为索引表示
            input：
                :param examples: 输入
                :param is_training:
            :return:
                features    Dict        特征
        """
        
    # 功能：生成数据
    def gen_data(self, file_path, is_training=True):
        """
            功能：生成数据
            input：
                :param file_path:
                :param is_training:
            :return:
                features
        """

    def gen_batch(self, batch_features):
        """
            功能：将batch中同一键对应的值组合在一起
            input:
                :param batch_features:
            :return:
                batch
        """

    def next_batch(self, features, is_training=True):
        """
          功能：生成batch数据
          input:
              :param features:
              :param is_training:
          :return:
              batch_data
        """

```

#### 5.1.2 生成数据主函数 gen_data()

##### 5.1.2.1  读取原始数据 模块

- 函数调用

```python
class TrainData(object):
  def gen_data(self, file_path, is_training=True):
    ...
    # 1，读取原始数据
    examples = self.read_data(file_path, is_training)
    print("read finished")
    ...
```

- read_data() 函数代码介绍：

```python
    # 功能：加载数据
    def read_data(self, file_path, is_training):
        """
            功能：加载数据
            input：
                file_path:        String       数据文件
                is_training       boolean      是否训练
            :return:
                examples
        """
        with open(file_path, 'r', encoding="utf8") as f:
            train_data = json.load(f)
            train_data = train_data['data']

        examples = []
        # 1, 遍历所有的训练数据，取出每一篇文章
        for article in train_data:
            # 2， 遍历每一篇文章，取出该文章下的所有段落
            for para in article['paragraphs']:
                context = para['context']  # 取出当前段落的内容
                # 将原文分割成列表返回，主要是确保一连串的数字，或者英文单词作为一个单独的token存在
                doc_tokens = self._split_char(context)

                # char_to_word_offset 的长度等于context的长度，但是列表中的最大值为len(doc_tokens) - 1
                # 主要作用是为了维护 doc_tokens 中的 token 的位置对应到在 context 中的位置
                char_to_word_offset = []
                for index, token in enumerate(doc_tokens):
                    for i in range(len(token)):
                        char_to_word_offset.append(index)

                # 把问答对读取出来
                for qas in para['qas']:
                    qid = qas['id']
                    ques_text = qas['question']
                    ans_text = qas['answers'][0]['text']

                    start_position_final = -1
                    end_position_final = -1
                    if is_training:

                        # 取出在原始context中的start和end position
                        start_position = qas['answers'][0]['answer_start']

                        # 按照答案长度取去计算结束位置
                        end_position = start_position + len(ans_text) - 1

                        # 如果在start的位置上是对应原始context中的空字符，则往上加一位
                        while context[start_position] == " " or context[start_position] == "\t" or \
                                context[start_position] == "\r" or context[start_position] == "\n":

                            start_position += 1

                        # 从context中start和end的位置映射到doc_tokens中的位置
                        start_position_final = char_to_word_offset[start_position]
                        end_position_final = char_to_word_offset[end_position]

                        if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".", ","}:
                            start_position_final += 1

                    if "".join(doc_tokens[start_position_final: end_position_final + 1]) != ans_text:
                        if ans_text != context[qas['answers'][0]['answer_start']: qas['answers'][0]['answer_start'] + len(ans_text)]:
                            print(qas["id"])
                            print(ans_text)
                            print(doc_tokens[start_position_final: end_position_final + 1])
                            print(context[qas['answers'][0]['answer_start']: qas['answers'][0]['answer_start'] + len(ans_text)])

                    examples.append({'doc_tokens': doc_tokens,
                                     'orig_answer_text': context,
                                     'qid': qid,
                                     'question': ques_text,
                                     'answer': ans_text,
                                     'start_position': start_position_final,
                                     'end_position': end_position_final})

        return examples
```

- 处理前数据：

```json
{
  "version": "v1.0", 
  "data": [
    {
      "paragraphs": [
        {
          "id": "TRAIN_186", 
          "context": "范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；... 范廷颂于2009年2月22日清晨在河内离世，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。", 
          "qas": [
            {
              "question": "范廷颂是什么时候被任为主教的？", 
              "id": "TRAIN_186_QUERY_0", 
              "answers": [
                {
                  "text": "1963年", 
                  "answer_start": 30
                }
              ]
            }, ...
          ]
        }
      ], 
      "id": "TRAIN_186", 
      "title": "范廷颂"
    }, ...
  ]
}
```

- 处理后数据：

```json
[
    {
        'doc_tokens': ['范', '廷', '颂', '枢', '机', '（', '，', '）', '，', '圣', '名', '保', '禄', '·', '若', '瑟', '（', '）', '，', '是', '越', '南', '罗', '马', '天', '主', '教', '枢', '机', '。', '1963', '年', '被', '任', '为', ' 主', '教', '；', '1990', '年', ... , '，', '享', '年', '89', '岁', '；', '其', '葬', '礼', '于', '同', '月', '26', '日', '上', '午', '在', '天', '主', '教', '河', '内', '总', '教', '区', '总', '主', '教', '座', '堂', '举', '行', '。'],
        'orig_answer_text': '范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；... 其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。',
        'qid': 'TRAIN_186_QUERY_0',
        'question': '范廷颂是什么时候被任为主教的？',
        'answer': '1963年',
        'start_position': 30,
        'end_position': 31
    }, 
    {
        'doc_tokens': ['范', '廷', '颂', '枢', '机', '（', '，', '）', '，', '圣', '名', '保', '禄', '·', '若', '瑟', '（', '）', '，', '是', '越', '南', '罗', '马', '天', '主', '教', '枢', '机', '。', '1963', '年', ..., '2009', '年', '2', '月', '22', '日', '清', '晨', '在', '河', '内', '离', '世', '，', '享', '年', '89', '岁', '；', '其', '葬', '礼', '于', '同', '月', '26', '日', '上', '午', '在', '天', '主', '教', '河', '内', '总', '教', '区', '总', '主', '教', '座', '堂', '举', '行', '。'],
        'orig_answer_text': '范廷颂枢机 （，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；... 范廷颂于2009年2月22日清晨在河内离世，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。',
        'qid': 'TRAIN_186_QUERY_1',
        'question': '1990年，范廷颂担任什么职务？',
        'answer': '1990年被擢升为天主教河内总教区宗座署理',
        'start_position': 38,
        'end_position': 55
    }, 
    {
        'doc_tokens': ['范', '廷', '颂', '枢', '机', '（', '，', '）', '，', '圣', '名', '保', '禄', '·', '若', '瑟', ' （', '）', '，', '是', '越', '南', '罗', '马', '天', '主', '教', '枢', '机', '。', '1963', '年', '被', '任', '为', '主', '教', '；', '1990', '年', ..., '2009', '年', '2', '月', '22', '日', '清', '晨', '在', '河', '内', '离', '世', '，', '享', '年', '89', '岁', '；', '其', '葬', '礼', '于', '同', '月', '26', '日', '上', '午', '在', '天', '主', '教', '河', '内', '总', '教', '区', '总', '主', '教', '座', '堂', '举', '行', '。'],
        'orig_answer_text': '范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；1990年...范廷颂于2009年2月22日清晨在河内离世，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。',
        'qid': 'TRAIN_186_QUERY_2',
        'question': '范廷颂是于何时何地出生的？',
        'answer': '范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生',
        'start_position': 85,
        'end_position': 109
    },...
]
```

- 函数思路

1. 加载 json 数据；
2. 遍历每一篇文章，取出该文章下的所有段落；
3. 对段落进行处理
   1. 将 段落 分割成列表返回；**【主要是确保一连串的数字，或者英文单词作为一个单独的token存在】**
   2. 利用 char_to_word_offset 做 段落到列表间的映射；**【主要作用是为了维护 doc_tokens 中的 token 的位置对应到在 context 中的位置】**

```s
  doc_tokens:['范', '廷', '颂', '枢', '机', '（', '，', '）', '，', '圣', '名', '保', '禄', '·', '若', '瑟', '（', '）', '，', '是', '越', '南', '罗', '马', '天', '主', '教', '枢', '机', '。', '1963', '年', ..., '堂', '举', '行', '。']
  char_to_word_offset:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 30, 30, 30, 31, ..., 745, 746, 747, 748]
```

> 注：在列表中 '1963' 对应 char_to_word_offset 中的位置为 30, 30, 30, 30，而 '年' 对应 31 

4. 对 问答对进行处理
   1. 读取 qid，question，answers；
   2. 取出在原始 context 中的 start 和 end position；
   3. 如果在start的位置上是对应原始 context 中的空字符，则往上加一位；
   4. 从 context 中 start 和 end 的位置映射到 doc_tokens 中的位置

```s
  origin start_position:30
  origin end_position:34
  start_position_final:30
  end_position_final:31
```

> 注：这里答案 “1963年” 在 原始 context 中的 start 位置为 30，end 位置为 34，但是在 doc_tokens 中，会被分成 ["1963","年"]，所以在 doc_tokens 对应的索引 为 start 位置为 30，end 位置为 31

##### 5.1.2.2 文本数据 编码为 特征 模块

- 函数调用

```python
class TrainData(object):
  def gen_data(self, file_path, is_training=True):
    ...
    # 2，输入转索引
    features = self.trans_to_features(examples, is_training)
    print("index transform finished")
    ...
```

- trans_to_features() 函数代码介绍：

```python
class TrainData(object):
  ...
  # 功能：将输入转化为索引表示
  def trans_to_features(self, examples, is_training):
      """
          功能：将输入转化为索引表示
          input：
              :param examples: 输入
              :param is_training:
          :return:
              features    Dict        特征
      """
      tokenizer = tokenization.FullTokenizer(vocab_file=self.__vocab_path, do_lower_case=True)
      features = []
      unique_id = 1000000000
      for (example_index, example) in enumerate(examples):
          # 用wordpiece的方法对query进行分词处理
          query_tokens = tokenizer.tokenize(example['question'])
          # 句子裁剪：给定query一个最大长度来控制query的长度
          if len(query_tokens) > self.__query_length:
              query_tokens = query_tokens[: self.__query_length]

          # 主要是针对context构造索引，之前我们将中文，标点符号，空格，一连串的数字，英文单词分割存储在doc_tokens中
          # 但在bert的分词器中会将一连串的数字，中文，英文等分割成子词，也就是说经过bert的分词之后得到的tokens和之前
          # 获得的doc_tokens是不一样的，因此我们仍需要对start和end position从doc_tokens中的位置映射到当前tokens的位置
          tok_to_orig_index = []  # 存储未分词的token的索引，但长度和下面的相等
          orig_to_tok_index = []  # 存储分词后的token的索引，但索引不是连续的，会存在跳跃的情况
          all_doc_tokens = []  # 存储分词后的token，理论上长度是要大于all_tokens的

          for (i, token) in enumerate(example['doc_tokens']):
              sub_tokens = tokenizer.tokenize(token)
              # orig_to_tok_index的长度等于doc_tokens，里面每个值存储的是doc_tokens中的token在all_doc_tokens中的起止索引值
              # 用来将在all_token中的start和end转移到all_doc_tokens中
              orig_to_tok_index.append([len(all_doc_tokens)])
              for sub_token in sub_tokens:
                  # tok_to_orig_index的长度等于all_doc_tokens, 里面会有重复的值
                  tok_to_orig_index.append(i)
                  all_doc_tokens.append(sub_token)
              orig_to_tok_index[-1].append(len(all_doc_tokens) - 1)

          tok_start_position = -1
          tok_end_position = -1
          if is_training:
              # 原来token到新token的映射，这是新token的起点
              tok_start_position = orig_to_tok_index[example['start_position']][0]
              tok_end_position = orig_to_tok_index[example['end_position']][1]

              tok_start_position, tok_end_position = self._improve_answer_span(
                  all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                  example['orig_answer_text'])

          # The -3 accounts for [CLS], [SEP] and [SEP]
          max_tokens_for_doc = self.__max_length - len(query_tokens) - 3

          doc_spans = []
          _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])

          # 在使用bert的时候，一般会将最大的序列长度控制在512，因此对于长度大于最大长度的context，我们需要将其分成多个片段
          # 采用滑窗的方式，滑窗大小是小于最大长度的，因此分割的片段之间是存在重复的子片段。
          start_offset = 0  # 截取的片段的起始位置
          while start_offset < len(all_doc_tokens):
              length = len(all_doc_tokens) - start_offset

              # 当长度超标，需要使用滑窗
              if length > max_tokens_for_doc:
                  length = max_tokens_for_doc
              doc_spans.append(_DocSpan(start=start_offset, length=length))
              if start_offset + length == len(all_doc_tokens):  # 当length < max_len时，该条件成立
                  break
              start_offset += min(length, self.__doc_stride)

          # 组合query和context的片段成一个序列输入到bert中
          for (doc_span_index, doc_span) in enumerate(doc_spans):
              tokens = []
              token_to_orig_map = {}
              # 因为片段之间会存在重复的子片段，但是子片段中的token在不同的片段中的重要性是不一样的，
              # 在这里根据上下文的数量来决定token的重要性，在之后预测时对于出现在两个片段中的token，
              # 只取重要性高的片段中的token的分数作为该token的分数
              token_is_max_context = {}
              segment_ids = []
              tokens.append("[CLS]")
              segment_ids.append(0)
              for token in query_tokens:
                  tokens.append(token)
                  segment_ids.append(0)
              tokens.append("[SEP]")
              segment_ids.append(0)

              for i in range(doc_span.length):
                  split_token_index = doc_span.start + i
                  # 映射当前span组成的句子对的索引到原始token的索引
                  token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                        

                  # 在利用滑窗分割多个span时会存在有的词出现在两个span中，但最后统计的时候，我们只能选择一个span，因此
                  # 作者根据该词上下文词的数量构建了一个分数，取分数最高的那个span
                  is_max_context = self._check_is_max_context(doc_spans, doc_span_index, split_token_index)
                  token_is_max_context[len(tokens)] = is_max_context
                  tokens.append(all_doc_tokens[split_token_index])
                  segment_ids.append(1)
              tokens.append("[SEP]")
              segment_ids.append(1)

              input_ids = tokenizer.convert_tokens_to_ids(tokens)

              # The mask has 1 for real tokens and 0 for padding tokens. Only real
              # tokens are attended to.
              input_mask = [1] * len(input_ids)

              # Zero-pad up to the sequence length.
              while len(input_ids) < self.__max_length:
                  input_ids.append(0)
                  input_mask.append(0)
                  segment_ids.append(0)

              assert len(input_ids) == self.__max_length
              assert len(input_mask) == self.__max_length
              assert len(segment_ids) == self.__max_length

              start_position = -1
              end_position = -1
              if is_training:
                  # For training, if our document chunk does not contain an annotation
                  # we throw it out, since there is nothing to predict.
                  if tok_start_position == -1 and tok_end_position == -1:
                      start_position = 0  # 问题本来没答案，0是[CLS]的位子
                      end_position = 0
                  else:  # 如果原本是有答案的，那么去除没有答案的feature
                      out_of_span = False
                      doc_start = doc_span.start  # 映射回原文的起点和终点
                      doc_end = doc_span.start + doc_span.length - 1
                      # 该划窗没答案作为无答案增强
                      if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  
                          out_of_span = True
                      if out_of_span:
                          start_position = 0
                          end_position = 0
                      else:
                          doc_offset = len(query_tokens) + 2
                          start_position = tok_start_position - doc_start + doc_offset
                          end_position = tok_end_position - doc_start + doc_offset

              features.append({'unique_id': unique_id,
                                'example_index': example_index,
                                'doc_span_index': doc_span_index,
                                'tokens': tokens,
                                'token_to_orig_map': token_to_orig_map,
                                'token_is_max_context': token_is_max_context,
                                'input_ids': input_ids,
                                'input_mask': input_mask,
                                'segment_ids': segment_ids,
                                'start_position': start_position,
                                'end_position': end_position})
              unique_id += 1
      return features
```

- 函数思路

1. 加载 Bert 词典
2. 用wordpiece的方法对 question 进行分词处理

```s
# 输入 => 输出
范廷颂是什么时候被任为主教的？ => ['范', '廷', '颂', '是', '什', '么', '时', '候', '被', '任', '为', '主', '教', '的', '？']
1990年，范廷颂担任什么职务？ => ['1990', '年', '，', '范', '廷', '颂', '担', '任', '什', '么', '职', '务', '？']
范廷颂是于何时何地出生的？ => ['范', '廷', '颂', '是', '于', '何', '时', '何', '地', '出', '生', '的', '？']
1994年3月，范廷颂担任什么职务？ => ['1994', '年', '3', '月', '，', '范', '廷', '颂', '担', '任', '什', '么', '职', '务', '？']
范廷颂是何时去世的？ => ['范', '廷', '颂', '是', '何', '时', '去', '世', '的', '？']
安雅·罗素法参加了什么比赛获得了亚军？ => ['安', '雅', '·', '罗', '素', '法', '参', '加', '了', '什', '么', '比', '赛', '获', '得', '了', '亚', '军', '？']
Russell Tanoue对安雅·罗素法的评价是什么？ => ['russell', 'tan', '##ou', '##e', '对', '安', '雅', '·', '罗', '素', '法', '的', '评', '价', '是', '什', '么', '？']
安雅·罗素法合作过的香港杂志有哪些？ => ['安', '雅', '·', '罗', '素', '法', '合', '作', '过', '的', '香', '港', '杂', '志', '有', '哪', '些', '？']
毕业后的安雅·罗素法职业是什么？ => ['毕', '业', '后', '的', '安', '雅', '·', '罗', '素', '法', '职', '业', '是', '什', '么', '？']
```

3. 句子裁剪：给定query一个最大长度来控制query的长度【这里 query_length 取 64】;
4. 对start和end  position从doc_tokens中的位置映射到当前tokens的位置 （注：主要是针对context构造索引，之前我们将中文，标点符号，空格，一连串的数字，英文单词分割存储在doc_tokens中，但在**bert的分词器中会将一连串的数字，中文，英文等分割成子词**，也就是说经过**bert的分词之后得到的tokens和之前获得的doc_tokens是不一样的**）
5. 如果 is_training 是 True，那么 需要 **返回与标注答案更好匹配的标记化答案范围**

```python
  # 如果 is_training 是 True，那么 需要 **返回与标注答案更好匹配的标记化答案范围**
  if is_training:
      # 原来token到新token的映射，这是新token的起点
      tok_start_position = orig_to_tok_index[example['start_position']][0]
      tok_end_position = orig_to_tok_index[example['end_position']][1]

      tok_start_position, tok_end_position = self._improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, tokenizer,example['orig_answer_text']
      )
```

> self._improve_answer_span
```python
# 功能：返回与标注答案更好匹配的标记化答案范围。
def _improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer,
                          orig_answer_text):
    """
        功能：返回与标注答案更好匹配的标记化答案范围。  
    """
    '''
        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
    '''
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)
```

6. 计算 doc 的最大长度，因为 需要考虑 que 的长度和 [CLS], [SEP] and [SEP]

```python
  max_tokens_for_doc = self.__max_length - len(query_tokens) - 3
```

7. 对于 query 长度 大于 512 的，需要使用 滑窗（注：**在使用bert的时候，一般会将最大的序列长度控制在512**，因此对于长度大于最大长度的context，我们需要将其分成多个片段采用滑窗的方式，滑窗大小是小于最大长度的，因此分割的片段之间是存在重复的子片段。）

```python
    def trans_to_features(self, examples, is_training):
        ...
            doc_spans = []
            _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
            start_offset = 0  # 截取的片段的起始位置
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                # 当长度超标，需要使用滑窗
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                # 当length < max_len时，该条件成立
                if start_offset + length == len(all_doc_tokens):  
                    break
                start_offset += min(length, self.__doc_stride)
        ...
    >>>
    doc_spans:[DocSpan(start=0, length=494), DocSpan(start=128, length=494), DocSpan(start=256, length=493)]
    doc_spans:[DocSpan(start=0, length=496), DocSpan(start=128, length=496), DocSpan(start=256, length=493)]
    doc_spans:[DocSpan(start=0, length=496), DocSpan(start=128, length=496), DocSpan(start=256, length=493)]
    doc_spans:[DocSpan(start=0, length=494), DocSpan(start=128, length=494), DocSpan(start=256, length=493)]
    doc_spans:[DocSpan(start=0, length=499), DocSpan(start=128, length=499), DocSpan(start=256, length=493)]
    doc_spans:[DocSpan(start=0, length=490), DocSpan(start=128, length=490), DocSpan(start=256, length=366)]
    doc_spans:[DocSpan(start=0, length=491), DocSpan(start=128, length=491), DocSpan(start=256, length=366)]
    doc_spans:[DocSpan(start=0, length=491), DocSpan(start=128, length=491), DocSpan(start=256, length=366)]
    doc_spans:[DocSpan(start=0, length=493), DocSpan(start=128, length=493), DocSpan(start=256, length=366)]
    ...
```

8. 组合 query 和 context 的片段成一个序列输入到 bert 中

```python
    def trans_to_features(self, examples, is_training):
        ...
            # 组合 query 和 context 的片段成一个序列输入到 bert 中
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                # 因为片段之间会存在重复的子片段，但是子片段中的token在不同的片段中的重要性是不一样的，在这里根据上下文的数量来决定token的重要性，在之后预测时对于出现在两个片段中的token，只取重要性高的片段中的token的分数作为该token的分数
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    # 映射当前span组成的句子对的索引到原始token的索引
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                          
                    # 在利用滑窗分割多个span时会存在有的词出现在两个span中，但最后统计的时候，我们只能选择一个span，因此，作者根据该词上下文词的数量构建了一个分数，取分数最高的那个span
                    is_max_context = self._check_is_max_context(doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < self.__max_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == self.__max_length
                assert len(input_mask) == self.__max_length
                assert len(segment_ids) == self.__max_length

                start_position = -1
                end_position = -1
                if is_training:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    if tok_start_position == -1 and tok_end_position == -1:
                        start_position = 0  # 问题本来没答案，0是[CLS]的位子
                        end_position = 0
                    else:  # 如果原本是有答案的，那么去除没有答案的feature
                        out_of_span = False
                        doc_start = doc_span.start  # 映射回原文的起点和终点
                        doc_end = doc_span.start + doc_span.length - 1
                        # 该划窗没答案作为无答案增强
                        if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  
                            out_of_span = True
                        if out_of_span:
                            start_position = 0
                            end_position = 0
                        else:
                            doc_offset = len(query_tokens) + 2
                            start_position = tok_start_position - doc_start + doc_offset
                            end_position = tok_end_position - doc_start + doc_offset

                features.append({'unique_id': unique_id,
                                 'example_index': example_index,
                                 'doc_span_index': doc_span_index,
                                 'tokens': tokens,
                                 'token_to_orig_map': token_to_orig_map,
                                 'token_is_max_context': token_is_max_context,
                                 'input_ids': input_ids,
                                 'input_mask': input_mask,
                                 'segment_ids': segment_ids,
                                 'start_position': start_position,
                                 'end_position': end_position})
                unique_id += 1
        return features
        ...
```

> 输出

```python
    [
        {
            'unique_id': 1000000021, 
            'example_index': 7, 
            'doc_span_index': 0, 
            'tokens': ['[CLS]', '安', '雅', '·', '罗', '素', '法', '合', '作', '过', '的', '香', '港', '杂', '志', '有', '哪', '些', '？', '[SEP]', '安', '雅', '·', '罗', '素', '法', '（', '，', '）', '，', '来', '自', '俄', '罗', '斯', '圣', '彼', '得', '堡', ' 的', '模', '特', '儿', '。', '她', '是', '《', '全', '美', '超', '级', '模', '特', '儿', '新', '秀', '大', '赛', '》', '第', '十', '季', '的', '亚', '军', '。', '2008', '年', '，', '安', '雅', '宣', '布', '改', '回', '出', '生', '时', '的', '名', '字', '：', '安', '雅', '·', '罗', '素', '法', '（', 'an', '##ya', 'ro', '##zo', '##va', '）', '，', '在', '此', '之', '前', '是', '使', '用', '安', '雅', '·', '冈', '（', '）', '。', '安', '雅', '于', '俄', '罗', '斯', '出', '生', '，', '后', '来', '被', '一', '个', '居', '住', '在', '美', '国', '夏', '威', '夷', '群', '岛', '欧', '胡', '岛', '檀', '香', '山', '的', '家', '庭', '领', '养', '。', '安', '雅', '十', '七', '岁', '时', '曾', '参', '与', '香', '奈', '儿', '、', '路', '易', '·', '威', '登', '及', '芬', '迪', '（', 'fe', '##ndi', '）', '等', '品', '牌', '的', '非', '正', ' 式', '时', '装', '秀', '。', '2007', '年', '，', '她', '于', '瓦', '伊', '帕', '胡', '高', '级', '中', '学', '毕', '业', '。', '毕', '业', '后', '，', '她', '当', '了', '一', '名', '售', '货', '员', '。', '她', '曾', '为', 'russell', 'tan', '##ou', '##e', '拍', '摄', '照', '片', '，', 'russell', 'tan', '##ou', '##e', '称', '赞', '她', '是', '「', '有', '前', '途', '的', '新', '面', '孔', '」', '。', '安', '雅', '在', '半', '准', '决', '赛', '面', '试', '时', '说', '她', '对', '模', '特', '儿', '行', '业', '充', '满', '热', '诚', '，', '所', '以', '参', '加', '全', '美', '超', '级', '模', '特', '儿', '新', '秀', '大', '赛', '。', '她', '于', '比', '赛', '中', '表', '现', '出', '色', '，', '曾', '五', '次', '首', '名', '入', '围', '，', '平', '均', '入', '围', '顺', '序', '更', '拿', '下', '历', '届', '以', '来', '最', '优', '异', '的', '成', '绩', '(', '2', '.', '64', ')', '，', '另', '外', '胜', '出', '三', '次', '小', '挑', '战', '，', '分', '别', '获', '得', '与', '评', '判', '尼', '祖', '·', '百', '克', '拍', '照', '、', '为', '柠', '檬', '味', '道', '的', '七', '喜', '拍', '摄', '广', '告', '的', '机', '会', '及', '十', '万', '美', '元', '、', '和', '盖', '马', '蒂', '洛', '（', 'ga', '##i', 'matt', '##io', '##lo', '）', '设', '计', '的', '晚', '装', '。', '在', '最', '后', '两', '强', '中', ' ，', '安', '雅', '与', '另', '一', '名', '参', '赛', '者', '惠', '妮', '·', '汤', '姆', '森', '为', '范', '思', '哲', '走', '秀', '，', '但', '评', '判', '认', '为', '她', '在', '台', '上', '不', '够', '惠', '妮', '突', '出', '，', '所', '以', '选', '了', '惠', '妮', '当', '冠', '军', '，', '安', '雅', '屈', '居', '亚', '军', '(', '但', '就', '整', '体', ' 表', '现', '来', '说', '，', '部', '份', '网', '友', '认', '为', '安', '雅', '才', '是', '第', '十', '季', '名', '副', '其', '实', '的', '冠', '军', '。', ')', '安', '雅', '在', '比', '赛', '拿', '五', '次', '第', '一', '，', '也', '胜', ' 出', '多', '次', '小', '挑', '战', '。', '安', '雅', '赛', '后', '再', '次', '与', 'russell', 'tan', '##ou', '##e', '[SEP]'], 
            'token_to_orig_map': {20: 0, 21: 1, 22: 2, 23: 3, 24: 4, 25: 5, 26: 6, 27: 7, 28: 8, 29: 9, 30: 10, 31: 11, 32: 12, 33: 13, 34: 14, 35: 15, 36: 16, 37: 17, 38: 18, 39: 19, 40: 20, 41: 21, 42: 22, 43: 23, 44: 24, 45: 25, 46: 26, 47: 27, 48: 28, 49: 29, 50: 30, 51: 31, 52: 32, 53: 33, 54: 34, 55: 35, 56: 36, 57: 37, 58: 38, 59: 39, 60: 40, 61: 41, 62: 42, 63: 43, 64: 44, 65: 45, 66: 46, 67: 47, 68: 48, 69: 49, 70: 50, 71: 51, 72: 52, 73: 53, 74: 54, 75: 55, 76: 56, 77: 57, 78: 58, 79: 59, 80: 60, 81: 61, 82: 62, 83: 63, 84: 64, 85: 65, 86: 66, 87: 67, 88: 68, 89: 69, 90: 69, 91: 71, 92: 71, 93: 71, 94: 72, 95: 73, 96: 74, 97: 75, 98: 76, 99: 77, 100: 78, 101: 79, 102: 80, 103: 81, 104: 82, 105: 83, 106: 84, 107: 85, 108: 86, 109: 87, 110: 88, 111: 89, 112: 90, 113: 91, 114: 92, 115: 93, 116: 94, 117: 95, 118: 96, 119: 97, 120: 98, 121: 99, 122: 100, 123: 101, 124: 102, 125: 103, 126: 104, 127: 105, 128: 106, 129: 107, 130: 108, 131: 109, 132: 110, 133: 111, 134: 112, 135: 113, 136: 114, 137: 115, 138: 116, 139: 117, 140: 118, 141: 119, 142: 120, 143: 121, 144: 122, 145: 123, 146: 124, 147: 125, 148: 126, 149: 127, 150: 128, 151: 129, 152: 130, 153: 131, 154: 132, 155: 133, 156: 134, 157: 135, 158: 136, 159: 137, 160: 138, 161: 139, 162: 140, 163: 141, 164: 142, 165: 143, 166: 144, 167: 145, 168: 146, 169: 146, 170: 147, 171: 148, 172: 149, 173: 150, 174: 151, 175: 152, 176: 153, 177: 154, 178: 155, 179: 156, 180: 157, 181: 158, 182: 159, 183: 160, 184: 161, 185: 162, 186: 163, 187: 164, 188: 165, 189: 166, 190: 167, 191: 168, 192: 169, 193: 170, 194: 171, 195: 172, 196: 173, 197: 174, 198: 175, 199: 176, 200: 177, 201: 178, 202: 179, 203: 180, 204: 181, 205: 182, 206: 183, 207: 184, 208: 185, 209: 186, 210: 187, 211: 188, 212: 189, 213: 190, 214: 191, 215: 193, 216: 193, 217: 193, 218: 194, 219: 195, 220: 196, 221: 197, 222: 198, 223: 199, 224: 201, 225: 201, 226: 201, 227: 202, 228: 203, 229: 204, 230: 205, 231: 206, 232: 207, 233: 208, 234: 209, 235: 210, 236: 211, 237: 212, 238: 213, 239: 214, 240: 215, 241: 216, 242: 217, 243: 218, 244: 219, 245: 220, 246: 221, 247: 222, 248: 223, 249: 224, 250: 225, 251: 226, 252: 227, 253: 228, 254: 229, 255: 230, 256: 231, 257: 232, 258: 233, 259: 234, 260: 235, 261: 236, 262: 237, 263: 238, 264: 239, 265: 240, 266: 241, 267: 242, 268: 243, 269: 244, 270: 245, 271: 246, 272: 247, 273: 248, 274: 249, 275: 250, 276: 251, 277: 252, 278: 253, 279: 254, 280: 255, 281: 256, 282: 257, 283: 258, 284: 259, 285: 260, 286: 261, 287: 262, 288: 263, 289: 264, 290: 265, 291: 266, 292: 267, 293: 268, 294: 269, 295: 270, 296: 271, 297: 272, 298: 273, 299: 274, 300: 275, 301: 276, 302: 277, 303: 278, 304: 279, 305: 280, 306: 281, 307: 282, 308: 283, 309: 284, 310: 285, 311: 286, 312: 287, 313: 288, 314: 289, 315: 290, 316: 291, 317: 292, 318: 293, 319: 294, 320: 295, 321: 296, 322: 297, 323: 298, 324: 299, 325: 300, 326: 301, 327: 302, 328: 303, 329: 304, 330: 305, 331: 306, 332: 307, 333: 308, 334: 309, 335: 310, 336: 311, 337: 312, 338: 313, 339: 314, 340: 315, 341: 316, 342: 317, 343: 318, 344: 319, 345: 320, 346: 321, 347: 322, 348: 323, 349: 324, 350: 325, 351: 326, 352: 327, 353: 328, 354: 329, 355: 330, 356: 331, 357: 332, 358: 333, 359: 334, 360: 335, 361: 336, 362: 337, 363: 338, 364: 339, 365: 340, 366: 341, 367: 342, 368: 343, 369: 344, 370: 345, 371: 346, 372: 347, 373: 348, 374: 349, 375: 350, 376: 350, 377: 352, 378: 352, 379: 352, 380: 353, 381: 354, 382: 355, 383: 356, 384: 357, 385: 358, 386: 359, 387: 360, 388: 361, 389: 362, 390: 363, 391: 364, 392: 365, 393: 366, 394: 367, 395: 368, 396: 369, 397: 370, 398: 371, 399: 372, 400: 373, 401: 374, 402: 375, 403: 376, 404: 377, 405: 378, 406: 379, 407: 380, 408: 381, 409: 382, 410: 383, 411: 384, 412: 385, 413: 386, 414: 387, 415: 388, 416: 389, 417: 390, 418: 391, 419: 392, 420: 393, 421: 394, 422: 395, 423: 396, 424: 397, 425: 398, 426: 399, 427: 400, 428: 401, 429: 402, 430: 403, 431: 404, 432: 405, 433: 406, 434: 407, 435: 408, 436: 409, 437: 410, 438: 411, 439: 412, 440: 413, 441: 414, 442: 415, 443: 416, 444: 417, 445: 418, 446: 419, 447: 420, 448: 421, 449: 422, 450: 423, 451: 424, 452: 425, 453: 426, 454: 427, 455: 428, 456: 429, 457: 430, 458: 431, 459: 432, 460: 433, 461: 434, 462: 435, 463: 436, 464: 437, 465: 438, 466: 439, 467: 440, 468: 441, 469: 442, 470: 443, 471: 444, 472: 445, 473: 446, 474: 447, 475: 448, 476: 449, 477: 450, 478: 451, 479: 452, 480: 453, 481: 454, 482: 455, 483: 456, 484: 457, 485: 458, 486: 459, 487: 460, 488: 461, 489: 462, 490: 463, 491: 464, 492: 465, 493: 466, 494: 467, 495: 468, 496: 469, 497: 470, 498: 471, 499: 472, 500: 473, 501: 474, 502: 475, 503: 476, 504: 477, 505: 478, 506: 479, 507: 480, 508: 482, 509: 482, 510: 482}, 
            'token_is_max_context': {20: True, 21: True, 22: True, 23: True, 24: True, 25: True, 26: True, 27: True, 28: True, 29: True, 30: True, 31: True, 32: True, 33: True, 34: True, 35: True, 36: True, 37: True, 38: True, 39: True, 40: True, 41: True, 42: True, 43: True, 44: True, 45: True, 46: True, 47: True, 48: True, 49: True, 50: True, 51: True, 52: True, 53: True, 54: True, 55: True, 56: True, 57: True, 58: True, 59: True, 60: True, 61: True, 62: True, 63: True, 64: True, 65: True, 66: True, 67: True, 68: True, 69: True, 70: True, 71: True, 72: True, 73: True, 74: True, 75: True, 76: True, 77: True, 78: True, 79: True, 80: True, 81: True, 82: True, 83: True, 84: True, 85: True, 86: True, 87: True, 88: True, 89: True, 90: True, 91: True, 92: True, 93: True, 94: True, 95: True, 96: True, 97: True, 98: True, 99: True, 100: True, 101: True, 102: True, 103: True, 104: True, 105: True, 106: True, 107: True, 108: True, 109: True, 110: True, 111: True, 112: True, 113: True, 114: True, 115: True, 116: True, 117: True, 118: True, 119: True, 120: True, 121: True, 122: True, 123: True, 124: True, 125: True, 126: True, 127: True, 128: True, 129: True, 130: True, 131: True, 132: True, 133: True, 134: True, 135: True, 136: True, 137: True, 138: True, 139: True, 140: True, 141: True, 142: True, 143: True, 144: True, 145: True, 146: True, 147: True, 148: True, 149: True, 150: True, 151: True, 152: True, 153: True, 154: True, 155: True, 156: True, 157: True, 158: True, 159: True, 160: True, 161: True, 162: True, 163: True, 164: True, 165: True, 166: True, 167: True, 168: True, 169: True, 170: True, 171: True, 172: True, 173: True, 174: True, 175: True, 176: True, 177: True, 178: True, 179: True, 180: True, 181: True, 182: True, 183: True, 184: True, 185: True, 186: True, 187: True, 188: True, 189: True, 190: True, 191: True, 192: True, 193: True, 194: True, 195: True, 196: True, 197: True, 198: True, 199: True, 200: True, 201: True, 202: True, 203: True, 204: True, 205: True, 206: True, 207: True, 208: True, 209: True, 210: True, 211: True, 212: True, 213: True, 214: True, 215: True, 216: True, 217: True, 218: True, 219: True, 220: True, 221: True, 222: True, 223: True, 224: True, 225: True, 226: True, 227: True, 228: True, 229: True, 230: True, 231: True, 232: True, 233: True, 234: True, 235: True, 236: True, 237: True, 238: True, 239: True, 240: True, 241: True, 242: True, 243: True, 244: True, 245: True, 246: True, 247: True, 248: True, 249: True, 250: True, 251: True, 252: True, 253: True, 254: True, 255: True, 256: True, 257: True, 258: True, 259: True, 260: True, 261: True, 262: True, 263: True, 264: True, 265: True, 266: True, 267: True, 268: True, 269: True, 270: True, 271: True, 272: True, 273: True, 274: True, 275: True, 276: True, 277: True, 278: True, 279: True, 280: True, 281: True, 282: True, 283: True, 284: True, 285: True, 286: True, 287: True, 288: True, 289: True, 290: True, 291: True, 292: True, 293: True, 294: True, 295: True, 296: True, 297: True, 298: True, 299: True, 300: True, 301: True, 302: True, 303: True, 304: True, 305: True, 306: True, 307: True, 308: True, 309: True, 310: True, 311: True, 312: True, 313: True, 314: True, 315: True, 316: True, 317: True, 318: True, 319: True, 320: True, 321: True, 322: True, 323: True, 324: True, 325: True, 326: True, 327: True, 328: True, 329: True, 330: False, 331: False, 332: False, 333: False, 334: False, 335: False, 336: False, 337: False, 338: False, 339: False, 340: False, 341: False, 342: False, 343: False, 344: False, 345: False, 346: False, 347: False, 348: False, 349: False, 350: False, 351: False, 352: False, 353: False, 354: False, 355: False, 356: False, 357: False, 358: False, 359: False, 360: False, 361: False, 362: False, 363: False, 364: False, 365: False, 366: False, 367: False, 368: False, 369: False, 370: False, 371: False, 372: False, 373: False, 374: False, 375: False, 376: False, 377: False, 378: False, 379: False, 380: False, 381: False, 382: False, 383: False, 384: False, 385: False, 386: False, 387: False, 388: False, 389: False, 390: False, 391: False, 392: False, 393: False, 394: False, 395: False, 396: False, 397: False, 398: False, 399: False, 400: False, 401: False, 402: False, 403: False, 404: False, 405: False, 406: False, 407: False, 408: False, 409: False, 410: False, 411: False, 412: False, 413: False, 414: False, 415: False, 416: False, 417: False, 418: False, 419: False, 420: False, 421: False, 422: False, 423: False, 424: False, 425: False, 426: False, 427: False, 428: False, 429: False, 430: False, 431: False, 432: False, 433: False, 434: False, 435: False, 436: False, 437: False, 438: False, 439: False, 440: False, 441: False, 442: False, 443: False, 444: False, 445: False, 446: False, 447: False, 448: False, 449: False, 450: False, 451: False, 452: False, 453: False, 454: False, 455: False, 456: False, 457: False, 458: False, 459: False, 460: False, 461: False, 462: False, 463: False, 464: False, 465: False, 466: False, 467: False, 468: False, 469: False, 470: False, 471: False, 472: False, 473: False, 474: False, 475: False, 476: False, 477: False, 478: False, 479: False, 480: False, 481: False, 482: False, 483: False, 484: False, 485: False, 486: False, 487: False, 488: False, 489: False, 490: False, 491: False, 492: False, 493: False, 494: False, 495: False, 496: False, 497: False, 498: False, 499: False, 500: False, 501: False, 502: False, 503: False, 504: False, 505: False, 506: False, 507: False, 508: False, 509: False, 510: False}, 
            'input_ids': [101, 2128, 7414, 185, 5384, 5162, 3791, 1394, 868, 6814, 4638, 7676, 3949, 3325, 2562, 3300, 1525, 763, 8043, 102, 2128, 7414, 185, 5384, 5162, 3791, 8020, 8024, 8021, 8024, 3341, 5632, 915, 5384, 3172, 1760, 2516, 2533, 1836, 4638, 3563, 4294, 1036, 511, 1961, 3221, 517, 1059, 5401, 6631, 5277, 3563, 4294, 1036, 3173, 4899, 1920, 6612, 518, 5018, 1282, 2108, 4638, 762, 1092, 511, 8182, 2399, 8024, 2128, 7414, 2146, 2357, 3121, 1726, 1139, 4495, 3198, 4638, 1399, 2099, 8038, 2128, 7414, 185, 5384, 5162, 3791, 8020, 9064, 8741, 12910, 10121, 8786, 8021, 8024, 1762, 3634, 722, 1184, 3221, 886, 4500, 2128, 7414, 185, 1082, 8020, 8021, 511, 2128, 7414, 754, 915, 5384, 3172, 1139, 4495, 8024, 1400, 3341, 6158, 671, 702, 2233, 857, 1762, 5401, 1744, 1909, 2014, 1929, 5408, 2270, 3616, 5529, 2270, 3589, 7676, 2255, 4638, 2157, 2431, 7566, 1075, 511, 2128, 7414, 1282, 673, 2259, 3198, 3295, 1346, 680, 7676, 1937, 1036, 510, 6662, 3211, 185, 2014, 4633, 1350, 5705, 6832, 8020, 12605, 11874, 8021, 5023, 1501, 4277, 4638, 7478, 3633, 2466, 3198, 6163, 4899, 511, 8201, 2399, 8024, 1961, 754, 4482, 823, 2364, 5529, 7770, 5277, 704, 2110, 3684, 689, 511, 3684, 689, 1400, 8024, 1961, 2496, 749, 671, 1399, 1545, 6573, 1447, 511, 1961, 3295, 711, 11514, 12886, 9857, 8154, 2864, 3029, 4212, 4275, 8024, 11514, 12886, 9857, 8154, 4917, 6614, 1961, 3221, 519, 3300, 1184, 6854, 4638, 3173, 7481, 2096, 520, 511, 2128, 7414, 1762, 1288, 1114, 1104, 6612, 7481, 6407, 3198, 6432, 1961, 2190, 3563, 4294, 1036, 6121, 689, 1041, 4007, 4178, 6411, 8024, 2792, 809, 1346, 1217, 1059, 5401, 6631, 5277, 3563, 4294, 1036, 3173, 4899, 1920, 6612, 511, 1961, 754, 3683, 6612, 704, 6134, 4385, 1139, 5682, 8024, 3295, 758, 3613, 7674, 1399, 1057, 1741, 8024, 2398, 1772, 1057, 1741, 7556, 2415, 3291, 2897, 678, 1325, 2237, 809, 3341, 3297, 831, 2460, 4638, 2768, 5327, 113, 123, 119, 8308, 114, 8024, 1369, 1912, 5526, 1139, 676, 3613, 2207, 2904, 2773, 8024, 1146, 1166, 5815, 2533, 680, 6397, 1161, 2225, 4862, 185, 4636, 1046, 2864, 4212, 510, 711, 3387, 3597, 1456, 6887, 4638, 673, 1599, 2864, 3029, 2408, 1440, 4638, 3322, 833, 1350, 1282, 674, 5401, 1039, 510, 1469, 4667, 7716, 5881, 3821, 8020, 11005, 8169, 12042, 8652, 8897, 8021, 6392, 6369, 4638, 3241, 6163, 511, 1762, 3297, 1400, 697, 2487, 704, 8024, 2128, 7414, 680, 1369, 671, 1399, 1346, 6612, 5442, 2669, 1984, 185, 3739, 1990, 3481, 711, 5745, 2590, 1528, 6624, 4899, 8024, 852, 6397, 1161, 6371, 711, 1961, 1762, 1378, 677, 679, 1916, 2669, 1984, 4960, 1139, 8024, 2792, 809, 6848, 749, 2669, 1984, 2496, 1094, 1092, 8024, 2128, 7414, 2235, 2233, 762, 1092, 113, 852, 2218, 3146, 860, 6134, 4385, 3341, 6432, 8024, 6956, 819, 5381, 1351, 6371, 711, 2128, 7414, 2798, 3221, 5018, 1282, 2108, 1399, 1199, 1071, 2141, 4638, 1094, 1092, 511, 114, 2128, 7414, 1762, 3683, 6612, 2897, 758, 3613, 5018, 671, 8024, 738, 5526, 1139, 1914, 3613, 2207, 2904, 2773, 511, 2128, 7414, 6612, 1400, 1086, 3613, 680, 11514, 12886, 9857, 8154, 102], 
            'input_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
            'segment_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
            'start_position': 0, 
            'end_position': 0
        },...
    ]
...
```

##### 5.1.2.3 数据存储 模块

这一步 主要 是 对 上面生成的 examples 和 features 进行存储。

```python
    # 功能：生成数据
    def gen_data(self, file_path, is_training=True):
        """
            功能：生成数据
            input：
                :param file_path:
                :param is_training:
            :return:
                features
        """
        ...
        # 3，数据存储
        if is_training:
            with open(os.path.join(self.__output_path, "train_examples.json"), "w", encoding="utf8") as fw:
                json.dump(examples, fw, indent=4, ensure_ascii=False)

            with open(os.path.join(self.__output_path, "train_features.json"), "w", encoding="utf8") as fw:
                json.dump(features, fw, indent=4, ensure_ascii=False)
            return features
        else:
            with open(os.path.join(self.__output_path, "dev_examples.json"), "w", encoding="utf8") as fw:
                json.dump(examples, fw, indent=4, ensure_ascii=False)

            with open(os.path.join(self.__output_path, "dev_features.json"), "w", encoding="utf8") as fw:
                json.dump(features, fw, indent=4, ensure_ascii=False)

            return (examples, features)
```

### 5.2  BertMachineReading 类 【model.py】

#### 5.2.1 整体结构

```python
class BertMachineReading(object):
    def __init__(self, config, is_training=True, num_train_step=None, num_warmup_step=None):
        ...

    # 功能：模型构建
    def built_model(self):
        '''
            功能：模型构建
        '''

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())

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
        ...

    def eval(self, sess, batch):
        """
            功能：验证模型
            input：
                :param sess: tf中的会话对象
                :param batch: batch数据
            :return: 预测结果
                start_logits     首指针概率
                end_logits       尾指针概率
        """
        ...

    def infer(self, sess, batch):
        """
            功能：预测新数据
            input：
                :param sess: tf中的会话对象
                :param batch: batch数据
            :return: 预测结果
                start_logits     首指针概率
                end_logits       尾指针概率
        """
        ...
```

#### 5.2.2 BertMachineReading 类 初始化

```python
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

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())
```

#### 5.2.3 BertMachineReading 类 模型构建

```python
class BertMachineReading(object):
    def __init__(self, config, is_training=True, num_train_step=None, num_warmup_step=None):
        ...

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
```

#### 5.2.4 BertMachineReading 类 模型训练

```python
class BertMachineReading(object):
    ...
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
```

#### 5.2.5 BertMachineReading 类 模型验证

```python
class BertMachineReading(object):
    ...
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
```

#### 5.2.6 BertMachineReading 类 模型预测

```python
class BertMachineReading(object):
    ...
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
```

### 5.3  Trainer 类 【trainer.py】

```python
class Trainer(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r") as fr:
            self.config = json.load(fr)
        self.__bert_checkpoint_path = os.path.join(self.config["bert_model_path"], "bert_model.ckpt")

        # 创建数据对象
        self.data_obj = self.load_data()
        # 加载训练数据集 及 对数据进行编码
        self.t_features = self.data_obj.gen_data(
            self.config["train_data"]
        )
        # 加载验证数据集 及 对数据进行编码
        self.e_examples, self.e_features = self.data_obj.gen_data(
            self.config["eval_data"], is_training=False
        )
        print("train data size: {}".format(len(self.t_features)))
        print("eval data size: {}".format(len(self.e_features)))

        num_train_steps = int(
            len(self.t_features) / self.config["batch_size"] * self.config["epochs"]
        )
        num_warmup_steps = int(num_train_steps * self.config["warmup_rate"])
        # 初始化模型对象
        self.model = self.create_model(num_train_steps, num_warmup_steps)

    # 功能：创建数据对象
    def load_data(self):
        """
            功能：创建数据对象
            :return:
                data_obj        TrainData 对象
        """
        # 生成训练集对象并生成训练数据
        data_obj = TrainData(self.config)
        return data_obj

    # 功能：根据config文件选择对应的模型，并初始化
    def create_model(self, num_train_step, num_warmup_step):
        """
            功能：根据config文件选择对应的模型，并初始化
            input:
                 num_train_step
                 num_warmup_step
            :return:
                model
        """
        model = BertMachineReading(
            config=self.config, 
            num_train_step=num_train_step, 
            num_warmup_step=num_warmup_step
        )
        return model

    # 功能：训练
    def train(self):
        with tf.Session() as sess:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, self.__bert_checkpoint_path)
            print("init bert model params")
            tf.train.init_from_checkpoint(self.__bert_checkpoint_path, assignment_map)
            print("init bert model params done")
            sess.run(tf.variables_initializer(tf.global_variables()))

            current_step = 0
            start = time.time()
            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in self.data_obj.next_batch(self.t_features):
                    loss, start_logits, end_logits = self.model.train(sess, batch)
                    # print("start: ", start_logits)
                    # print("end: ", end_logits)
                    end = time.time()
                    tl = time.localtime(end)
                    print("train: time: {}, step: {}, loss: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", tl) , current_step, loss))
                    
                    current_step += 1
                    if self.data_obj and current_step % self.config["checkpoint_every"] == 0:

                        all_results = []
                        for eval_batch in self.data_obj.next_batch(self.e_features, is_training=False):
                            start_logits, end_logits = self.model.eval(sess, eval_batch)

                            for unique_id, start_logit, end_logit in zip(
                                eval_batch["unique_id"],start_logits,end_logits
                            ):
                                all_results.append(
                                    dict(
                                        unique_id=unique_id,
                                        start_logits=start_logit.tolist(),
                                        end_logits=end_logit.tolist()
                                    )
                                )

                        with open("output/cmrc2018/results.json", "w", encoding="utf8") as fw:
                            json.dump(all_results, fw, indent=4, ensure_ascii=False)

                        write_predictions(all_examples=self.e_examples,
                                          all_features=self.e_features,
                                          all_results=all_results,
                                          n_best_size=self.config["n_best_size"],
                                          max_answer_length=self.config["max_answer_length"],
                                          output_prediction_file=self.config["output_predictions_path"],
                                          output_nbest_file=self.config["output_nbest_path"])

                        result = get_eval(original_file=self.config["eval_data"],
                                          prediction_file=self.config["output_predictions_path"])

                        print("\n")
                        end = time.time()
                        tl = time.localtime(end)
                        print("eval:  time: {}, step: {}, f1: {}, em: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", tl) , current_step, result["f1"], result["em"]))
                        print("\n")

                        if self.config["ckpt_model_path"]:
                            save_path = self.config["ckpt_model_path"]
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)

            end = time.time()
            print("total train time: ", end - start)
```

