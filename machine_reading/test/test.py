# encoding=utf-8
import os
import json
import random
import collections
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from bert import tokenization


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

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
        """
            Check if this is the 'max context' doc span for the token.
        """
        '''
            # Because of the sliding window approach taken to scoring documents, a single
            # token can appear in multiple documents. E.g.
            #  Doc: the man went to the store and bought a gallon of milk
            #  Span A: the man went to the
            #  Span B: to the store and bought
            #  Span C: and bought a gallon of
            #  ...
            #
            # Now the word 'bought' will have two scores from spans B and C. We only
            # want to consider the score with "maximum context", which we define as
            # the *minimum* of its left and right context (the *sum* of left and
            # right context will always be the same, of course).
            #
            # In the example the maximum context for 'bought' would be span C since
            # it has 1 left context and 3 right context, while span B has 4 left context
            # and 0 right context.
        '''
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

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
        new_context = []
        pre_is_digit = False
        pre_is_letter = False
        for char in context:
            if "0" <= char <= "9":
                if pre_is_digit:
                    new_context[-1] += char
                else:
                    new_context.append(char)
                    pre_is_digit = True
                    pre_is_letter = False
            elif "a" <= char <= "z" or "A" <= char <= "Z":

                if pre_is_letter:
                    new_context[-1] += char
                else:
                    new_context.append(char)
                    pre_is_letter = True
                    pre_is_digit = False
            else:
                new_context.append(char)
                pre_is_digit = False
                pre_is_letter = False
        return new_context

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
        train_data = train_data[0:2]
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
        # 加载 Bert 词典
        tokenizer = tokenization.FullTokenizer(
          vocab_file=self.__vocab_path, do_lower_case=True
        )
        features = []
        unique_id = 1000000000
        for (example_index, example) in enumerate(examples):
            # 用wordpiece的方法对 question 进行分词处理
            query_tokens = tokenizer.tokenize(example['question'])

            # 句子裁剪：给定query一个最大长度来控制query的长度
            if len(query_tokens) > self.__query_length:
                query_tokens = query_tokens[: self.__query_length]

            '''
              对 start 和 end  position 从 doc_tokens 中的位置映射到当前 tokens 的位置

              解释：主要是针对context构造索引，之前我们将中文，标点符号，空格，一连串的数字，英文单词分割存储在doc_tokens中，但在bert的分词器中会将一连串的数字，中文，英文等分割成子词，也就是说经过bert的分词之后得到的tokens和之前获得的doc_tokens是不一样的
            '''
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
            
            # 如果 is_training 是 True，那么 需要 **返回与标注答案更好匹配的标记化答案范围**
            if is_training:
                # 原来token到新token的映射，这是新token的起点
                tok_start_position = orig_to_tok_index[example['start_position']][0]
                tok_end_position = orig_to_tok_index[example['end_position']][1]

                tok_start_position, tok_end_position = self._improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,example['orig_answer_text']
                )

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = self.__max_length - len(query_tokens) - 3

            '''
              对于 query 长度 大于 512 的，需要使用 滑窗

              在使用bert的时候，一般会将最大的序列长度控制在512，因此对于长度大于最大长度的context，我们需要将其分成多个片段采用滑窗的方式，滑窗大小是小于最大长度的，因此分割的片段之间是存在重复的子片段。
            '''
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

                print(f"features:{features}")
                unique_id += 1
        return features
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
        # 1，读取原始数据
        examples = self.read_data(file_path, is_training)
        print("read finished")

        # 2，输入转索引
        features = self.trans_to_features(examples, is_training)
        print("index transform finished")

        sys.exit(0)

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

    def gen_batch(self, batch_features):
        """
            功能：将batch中同一键对应的值组合在一起
            input:
                :param batch_features:
            :return:
                batch
        """
        unique_id = []
        input_ids = []
        input_masks = []
        segment_ids = []
        start_position = []
        end_position = []
        for feature in batch_features:
            unique_id.append(feature["unique_id"])
            input_ids.append(feature["input_ids"])
            input_masks.append(feature["input_mask"])
            segment_ids.append(feature["segment_ids"])
            start_position.append(feature["start_position"])
            end_position.append(feature["end_position"])

        return dict(
            unique_id=unique_id,
            input_ids=input_ids,
            input_masks=input_masks,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position
        )

    def next_batch(self, features, is_training=True):
        """
            功能：生成batch数据
            input:
                :param features:
                :param is_training:
            :return:
                batch_data
        """
        if is_training:
            random.shuffle(features)

        num_batches = len(features) // self.__batch_size
        if not is_training and (num_batches * self.__batch_size) < len(features):
            num_batches += 1

        for i in range(num_batches):
            start = i * self.__batch_size
            end = start + self.__batch_size
            batch_features = features[start: end]
            batch_data = self.gen_batch(batch_features)

            yield batch_data



config_path = 'cmrc_config_test.json'
with open(config_path, "r") as fr:
  config = json.load(fr)

train_data = {
  "version": "v1.0", 
  "data": [
    {
      "paragraphs": [
        {
          "id": "TRAIN_186", 
          "context": "范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；1994年被擢升为总主教，同年年底被擢升为枢机；2009年2月离世。范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生；童年时接受良好教育后，被一位越南神父带到河内继续其学业。范廷颂于1940年在河内大修道院完成神学学业。范廷颂于1949年6月6日在河内的主教座堂晋铎；及后被派到圣女小德兰孤儿院服务。1950年代，范廷颂在河内堂区创建移民接待中心以收容到河内避战的难民。1954年，法越战争结束，越南民主共和国建都河内，当时很多天主教神职人员逃至越南的南方，但范廷颂仍然留在河内。翌年管理圣若望小修院；惟在1960年因捍卫修院的自由、自治及拒绝政府在修院设政治课的要求而被捕。1963年4月5日，教宗任命范廷颂为天主教北宁教区主教，同年8月15日就任；其牧铭为「我信天主的爱」。由于范廷颂被越南政府软禁差不多30年，因此他无法到所属堂区进行牧灵工作而专注研读等工作。范廷颂除了面对战争、贫困、被当局迫害天主教会等问题外，也秘密恢复修院、创建女修会团体等。1990年，教宗若望保禄二世在同年6月18日擢升范廷颂为天主教河内总教区宗座署理以填补该教区总主教的空缺。1994年3月23日，范廷颂被教宗若望保禄二世擢升为天主教河内总教区总主教并兼天主教谅山教区宗座署理；同年11月26日，若望保禄二世擢升范廷颂为枢机。范廷颂在1995年至2001年期间出任天主教越南主教团主席。2003年4月26日，教宗若望保禄二世任命天主教谅山教区兼天主教高平教区吴光杰主教为天主教河内总教区署理主教；及至2005年2月19日，范廷颂因获批辞去总主教职务而荣休；吴光杰同日真除天主教河内总教区总主教职务。范廷颂于2009年2月22日清晨在河内离世，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。", 
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
            }, 
            {
              "question": "1990年，范廷颂担任什么职务？", 
              "id": "TRAIN_186_QUERY_1", 
              "answers": [
                {
                  "text": "1990年被擢升为天主教河内总教区宗座署理", 
                  "answer_start": 41
                }
              ]
            }, 
            {
              "question": "范廷颂是于何时何地出生的？", 
              "id": "TRAIN_186_QUERY_2", 
              "answers": [
                {
                  "text": "范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生", 
                  "answer_start": 97
                }
              ]
            }, 
            {
              "question": "1994年3月，范廷颂担任什么职务？", 
              "id": "TRAIN_186_QUERY_3", 
              "answers": [
                {
                  "text": "1994年3月23日，范廷颂被教宗若望保禄二世擢升为天主教河内总教区总主教并兼天主教谅山教区宗座署理", 
                  "answer_start": 548
                }
              ]
            }, 
            {
              "question": "范廷颂是何时去世的？", 
              "id": "TRAIN_186_QUERY_4", 
              "answers": [
                {
                  "text": "范廷颂于2009年2月22日清晨在河内离世", 
                  "answer_start": 759
                }
              ]
            }
          ]
        }
      ], 
      "id": "TRAIN_186", 
      "title": "范廷颂"
    }, 
    {
      "paragraphs": [
        {
          "id": "TRAIN_54", 
          "context": "安雅·罗素法（，），来自俄罗斯圣彼得堡的模特儿。她是《全美超级模特儿新秀大赛》第十季的亚军。2008年，安雅宣布改回出生时的名字：安雅·罗素法（Anya Rozova），在此之前是使用安雅·冈（）。安雅于俄罗斯出生，后来被一个居住在美国夏威夷群岛欧胡岛檀香山的家庭领养。安雅十七岁时曾参与香奈儿、路易·威登及芬迪（Fendi）等品牌的非正式时装秀。2007年，她于瓦伊帕胡高级中学毕业。毕业后，她当了一名售货员。她曾为Russell Tanoue拍摄照片，Russell Tanoue称赞她是「有前途的新面孔」。安雅在半准决赛面试时说她对模特儿行业充满热诚，所以参加全美超级模特儿新秀大赛。她于比赛中表现出色，曾五次首名入围，平均入围顺序更拿下历届以来最优异的成绩(2.64)，另外胜出三次小挑战，分别获得与评判尼祖·百克拍照、为柠檬味道的七喜拍摄广告的机会及十万美元、和盖马蒂洛（Gai Mattiolo）设计的晚装。在最后两强中，安雅与另一名参赛者惠妮·汤姆森为范思哲走秀，但评判认为她在台上不够惠妮突出，所以选了惠妮当冠军，安雅屈居亚军(但就整体表现来说，部份网友认为安雅才是第十季名副其实的冠军。)安雅在比赛拿五次第一，也胜出多次小挑战。安雅赛后再次与Russell Tanoue合作，为2008年4月30日出版的MidWeek杂志拍摄封面及内页照。其后她参加了V杂志与Supreme模特儿公司合办的模特儿选拔赛2008。她其后更与Elite签约。最近她与香港的模特儿公司 Style International Management 签约，并在香港发展其模特儿事业。她曾在很多香港的时装杂志中任模特儿，《Jet》、《东方日报》、《Elle》等。", 
          "qas": [
            {
              "question": "安雅·罗素法参加了什么比赛获得了亚军？", 
              "id": "TRAIN_54_QUERY_0", 
              "answers": [
                {
                  "text": "《全美超级模特儿新秀大赛》第十季", 
                  "answer_start": 26
                }
              ]
            }, 
            {
              "question": "Russell Tanoue对安雅·罗素法的评价是什么？", 
              "id": "TRAIN_54_QUERY_1", 
              "answers": [
                {
                  "text": "有前途的新面孔", 
                  "answer_start": 247
                }
              ]
            }, 
            {
              "question": "安雅·罗素法合作过的香港杂志有哪些？", 
              "id": "TRAIN_54_QUERY_2", 
              "answers": [
                {
                  "text": "《Jet》、《东方日报》、《Elle》等", 
                  "answer_start": 706
                }
              ]
            }, 
            {
              "question": "毕业后的安雅·罗素法职业是什么？", 
              "id": "TRAIN_54_QUERY_3", 
              "answers": [
                {
                  "text": "售货员", 
                  "answer_start": 202
                }
              ]
            }
          ]
        }
      ], 
      "id": "TRAIN_54", 
      "title": "安雅·罗素法"
    }
  ]
}

trainData = TrainData(config)

examples = trainData.gen_data(config["train_data"])
print(f"examples:{examples}")