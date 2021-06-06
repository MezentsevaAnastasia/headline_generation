from nltk import word_tokenize
from typing import List
from rouge_metric import PyRouge


class Evaluator:

    def __init__(self):
        pass

    def get_rouge(self, hypotheses: List[str], references: List[str]):
        """
        :param: hypotheses - list of documents not tokenized
        references - List[doc1[ref1[sent1[words]], ref2], doc2[ref1, ref2]]
        :return:
        """
        rouge = PyRouge(rouge_n=(1, 2, 3), rouge_l=True, rouge_w=False, skip_gap=4)
        scores = rouge.evaluate_tokenized(self._tokenize(hypotheses), self._tokenize(references))
        return scores

    def _tokenize(self, document):
        tokenized_sents = list()
        for sent in document:
            tokenized_sents.append(word_tokenize(sent, language="russian"))
        return tokenized_sents
