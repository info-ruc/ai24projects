import math

def get_weight(freq, base):
    return base / math.sqrt(freq)

def get_preprocessing_function(
        tokenizer,
        article_tokenizer,
        sentence1_key,
        sentence2_key,
        similarity_key,
        padding,
        max_seq_length,
        model_args,
        use_weight=False,
        *args,
        ):
    def preprocess_function(examples):
        sent1_args = (examples[sentence1_key], )
        sent1_result = tokenizer(*sent1_args, padding=padding, max_length=max_seq_length, truncation=True)
        sent2_args = (examples[sentence2_key], )
        sent2_result = tokenizer(*sent2_args, padding=padding, max_length=max_seq_length, truncation=True)
        
        sent1_result['input_ids_2'] = sent2_result['input_ids']
        sent1_result['attention_mask_2'] = sent2_result['attention_mask']
        
        if 'token_type_ids' in sent2_result:
            sent1_result['token_type_ids_2'] = sent2_result['token_type_ids']
        if 'position_ids' in sent2_result:
            sent1_result['position_ids_2'] = sent2_result['position_ids']
        
        if similarity_key is None:
            sent1_result['labels'] = [1] * len(sent1_args[0])
        else:
            sent1_result['labels'] = examples[similarity_key]

        sent1_result["article_ids"] = article_tokenizer[examples[sentence2_key]] 

        if use_weight: 
            sent1_result["weights"] = [get_weight(article_tokenizer.freqs[idx], 
                                                  article_tokenizer.avg_freq)
                                        for idx in sent1_result["article_ids"]]

        return sent1_result

    return preprocess_function

# def get_preprocessing_function(
#         tokenizer,
#         sentence1_key,
#         sentence2_key,
#         similarity_key,
#         source_key,
#         article_bias,
#         padding,
#         max_seq_length,
#         model_args,
#         is_extract=False,
#         *args,
#         ):
#     def preprocess_function(examples):
#         sent1_args = (examples[sentence1_key], )
#         sent1_result = tokenizer(*sent1_args, padding=padding, max_length=max_seq_length, truncation=True)
#         sent2_args = (examples[sentence2_key], )
#         sent2_result = tokenizer(*sent2_args, padding=padding, max_length=max_seq_length, truncation=True)
        
#         sent1_result['input_ids_2'] = sent2_result['input_ids']
#         sent1_result['attention_mask_2'] = sent2_result['attention_mask']
        
#         if 'token_type_ids' in sent2_result:
#             sent1_result['token_type_ids_2'] = sent2_result['token_type_ids']
#         if 'position_ids' in sent2_result:
#             sent1_result['position_ids_2'] = sent2_result['position_ids']
        
#         if similarity_key is None:
#             sent1_result['labels'] = torch.ones(len(sent1_args[0]))
#         else:
#             sent1_result['labels'] = examples[similarity_key]

#         if is_extract:
#             import re
#             import cn2an
#             pattern = r"第(.+?)条"
#             texts = examples[sentence2_key]
#             sources = examples[source_key]
#             ids = []
#             for text, source in zip(texts, sources):
#                 # 使用正则表达式搜索文本
#                 match = re.search(pattern, text)
#                 article_id = match.group(1)
#                 id = cn2an.cn2an(article_id, "smart")
#                 ids.append(id + article_bias[source])

#             sent1_result["article_ids"] = ids               

#         return sent1_result

#     return preprocess_function

