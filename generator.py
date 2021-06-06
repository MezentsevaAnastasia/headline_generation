import pickle
from typing import List
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from utils import load_config
from evaluator import Evaluator
from loader import DatasetLoader


class Generator:

    def __init__(self):
        pass

    def get_title(self, sentences: List[str], vecs) -> str:
        return self._rank_sents(sentences, vecs)

    def _rank_sents(self, sentences: List[str], vecs):
        n_clusters = int(np.ceil(len(sentences)*0.4))
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans = kmeans.fit(vecs)
        avg = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, vecs)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        return sentences[closest[ordering[0]]]


if __name__ == '__main__':
    config = load_config('config.json')
    loader = DatasetLoader(config['corpus_path'])
    refs = loader.load_data(only_headlines=True)
    with open(config['vectors_path'], 'rb') as vec_f:
        vectors = pickle.load(vec_f)
    texts = loader.load_data(preprocess=True, only_texts=True)
    hypos = list()
    for i, text in enumerate(tqdm(texts)):
        hypos.append(Generator().get_title(text, vectors[i]))
    print(len(hypos))
    print(len(refs))
    print(Evaluator().get_rouge(hypos, refs))


"""
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

    def _make_sim_mat(self, text_length: int, vecs):
        sim_mat = np.zeros((text_length, text_length))
        for i in range(text_length):
            for j in range(text_length):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(vecs[i].reshape(1, 512), vecs[j].reshape(1, 512))[0, 0]
        return sim_mat

    def _rank_sents(self, sentences: List[str], vecs):
        nx_graph = nx.from_numpy_array(self._make_sim_mat(len(sentences), vecs))
        scores = nx.pagerank(nx_graph, max_iter=10000000000)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        return ranked_sentences[0][1]
"""