import json
import re
from tqdm import tqdm
from nltk import sent_tokenize


class DatasetLoader:

    def __init__(self, path):
        self._path = path
        self._regex = ['(&nbsp;|&ndash;|  )', '<[^<]+?>']

    def load_data(self, preprocess=False, only_texts=False, only_headlines=False):
        result = list()
        with open(self._path, 'rb') as f:
            for line in tqdm(f):
                document = json.loads(line)
                title = document['title'].lower()
                result_text = document['text'].lower()
                if preprocess:
                    # clean html tags
                    clean_text = re.sub(self._regex[1], ' ', result_text)
                    # clean useless symbols
                    result_text = re.sub(self._regex[0], ' ', clean_text)
                    result_text = result_text.strip()
                if only_headlines:
                    result.append(title)
                elif only_texts:
                    sentences = sent_tokenize(result_text, language="russian")
                    result.append(sentences)
                else:
                    result.append({'text': result_text, 'title': title})
        return result
