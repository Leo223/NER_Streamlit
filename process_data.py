import json
import pandas as pd

from transformers import BertForTokenClassification, BertTokenizerFast
from torch import LongTensor, device, cuda


class Ner(object):

    def __init__(self):
        self.model_path = './NER_model'

        self.device = device('cuda') if cuda.is_available() else device('cpu')

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

        # variables
        self.model = None
        self.id2tag, self.tag2id = None, None

    def load_model(self):

        self.model = BertForTokenClassification.from_pretrained(self.model_path).to(self.device)

        with open('{}/config.json'.format(self.model_path), 'r') as fp:
            config_dic = json.load(fp)

        self.id2tag = {k: config_dic.get('id2label').get(k) for k in config_dic.get('id2label')}
        self.tag2id = config_dic.get('label2id')

    def get_prediction(self, sample, return_wordpiece=False, split_into_words=False):
        tokenize = self.tokenizer(sample, is_split_into_words=split_into_words)
        tokens = LongTensor([tokenize.get('input_ids')]).to(self.device)
        if len(tokens[0]) > 512:
            return False
        attention_mask = LongTensor([tokenize.get('attention_mask')]).to(self.device)
        predictions = self.model.forward(input_ids=tokens,
                                         attention_mask=attention_mask)[0].argmax(axis=2).tolist()[0]

        cols = ['tokens', 'tags']
        tokens_full = tokenize.encodings[0].tokens

        if return_wordpiece:
            labels = [self.id2tag.get(_pred) for _pred in predictions]
            df_result = pd.DataFrame([tokens_full, labels], index=cols).T
        else:
            new_tokens, new_labels = [], []
            for token, label_idx in zip(tokens_full, predictions):
                if token.startswith("##"):
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
                    new_labels.append(self.id2tag[str(label_idx)])
                    new_tokens.append(token)

            df_result = pd.DataFrame([new_tokens[1:-1], new_labels[1:-1]], index=cols).T

        return df_result


def display_format(df):
    text_tagged = list()
    for _data, _tag in df.values:
        elem = ' {}'.format(_data)
        if _tag != 'O':
            text_tagged.append((elem, _tag, "#8ef"))
        else:
            text_tagged.append(elem)
    return text_tagged

