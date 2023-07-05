# Author: Laura Galera Alfaro

"""
Functions for generating ranking explanations
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import time
import copy
import math
from scipy.stats import rankdata
from sklearn.linear_model import RidgeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score
from pyDOE2 import lhs
from scipy.stats.distributions import norm


_model_yahoo = "/output_yahoo/export/latest_model/1684750780"
_test_yahoo = "/datasets/yahoo/test_yahoo.csv"

_name_features = [str(i + 1) for i in range(0, 100)]

_loaded_model = tf.saved_model.load(_model_yahoo)

def _float_feature(value):
    """Converts a numerical value into a TensorFlow Feature object of type FloatList"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Converts a numerical value into a TensorFlow Feature object of type Int64List"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_all(examples):
    """Converts a dataframe of documents into a list of TensorFlow Example"""
    list_examples = []
    for idx, row in examples.iterrows():
        example_dict = {
            f'{feat_name}': _float_feature(feat_val) for
            feat_name, feat_val in zip(_name_features, row.iloc[2:].tolist())
        }

        example_dict['relevance_label'] = _int64_feature(int(row['relevance_label']))

        example_proto = tf.train.Example(features=tf.train.Features(feature=example_dict))
        list_examples.append(example_proto.SerializeToString())

    return list_examples

class Sampling_generator:
    """Creates a set of syntethic documents around the doument to explain using a sampling technique
    (smote, gaussian, lhs, dlime) and statistical information about the whole set of real documents
    """

    def __init__(self, data, sample_size, strategy_name, k = 10):
        """Init.

        Args:
            data: pandas dataframe where each row is a document.
            sample_size: number of samples (integer)
            strategy_name: sampling strategy (string). Can be 'smote',
            'gaussian', 'lhs' or 'dlime'
            k: number of neighbors for smote sampling (integer)
        """
        self.data = data
        self.sample_size = sample_size
        self.k = k

        if strategy_name == 'smote':
            self.technique = self.smote_sampling
        elif strategy_name == 'gaussian':
            self.technique = self.gaussian_inverse_sampling
        elif strategy_name == 'lhs':
            self.technique = self.latin_hypercube_sampling
        elif strategy_name == 'dlime':
            self.technique = self.dlime_sampling
        else:
            raise ValueError('''Unknown strategy for sampling''')
    def num_clust(self, points):
        """Chooses the optimal number of clusters

        Args:
            points: array-like, shape (n_samples, n_features)
        Returns:
            An integer indicating the number of clusters.
            Min 2 and max 10.
        """
        silhouette_scores = []

        for n_clusters in range(2, 11):
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(points)
            silhouette_scores.append(silhouette_score(points, labels))

        max_score_index = np.argmax(silhouette_scores)
        cluster_count = max_score_index + 2

        return cluster_count

    def smote_sampling(self, instance_explained, qid_data):
        """Generates a batch of syntethic samples around the instance's neighborhood

        Args:
            instance_explained: document in qid_data to explain
            qid_data: pandas series corresponding to
            the documents returned by a specific query
        Returns:
            A list of documents (pandas series) of length
            sample_size
        """
        generated_docs = []

        nbrs = NearestNeighbors(n_neighbors= self.k + 1, metric='euclidean', algorithm='ball_tree')
        nbrs.fit(qid_data.iloc[:, 2:])
        instance = instance_explained.values[2:].astype(np.float32)
        distance, indices = nbrs.kneighbors(instance.reshape(1, -1))
        indices = indices.flatten()[1:] #ignore first neighbor

        for t in range(0, self.sample_size):
            new_instance_explained = copy.copy(instance_explained)
            indx_neighbor = np.random.choice(indices)
            random_neighbor = qid_data.iloc[indx_neighbor]

            for sel_feat in range(2, qid_data.shape[1]):
                diff = random_neighbor.iloc[sel_feat] - instance_explained.iloc[sel_feat]
                perturbation = instance_explained.iloc[sel_feat] + np.random.uniform() * diff
                new_instance_explained[sel_feat] = perturbation

            generated_docs.append(new_instance_explained)

        return generated_docs

    def latin_hypercube_sampling(self, instance_explained, qid_data):
        """Generates a batch of syntethic samples implementing latin hypercube sampling

        Args:
            instance_explained: document in qid_data to explain
            qid_data: pandas series corresponding to
            the documents returned by a specific query
        Returns:
            A list of documents (pandas series) of length
            sample_size
        """
        num_features = qid_data.shape[1] - 2
        lhs_data = lhs(num_features, samples=self.sample_size).reshape(self.sample_size, num_features)
        means = np.zeros(num_features)
        stdvs = np.array([1] * num_features)

        for i in range(num_features):
            lhs_data[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(lhs_data[:, i])

        lhs_data = np.array(lhs_data)
        generated_docs = []

        for t in range(lhs_data.shape[0]):
            new_instance_explained = copy.copy(instance_explained)

            for sel_feat in range(lhs_data.shape[1]):
                scale = np.std(self.data.iloc[:, sel_feat + 2].values)
                new_instance_explained[sel_feat + 2] = lhs_data[t][sel_feat] * scale + instance_explained[sel_feat + 2]

            generated_docs.append(new_instance_explained)

        return generated_docs

    def gaussian_inverse_sampling(self, instance_explained, qid_data):
        """Generates a batch of syntethic samples implementing gaussian sampling

        Args:
            instance_explained: document in qid_data to explain
            qid_data: pandas series corresponding to
            the documents returned by a specific query
        Returns:
            A list of documents (pandas series) of length
            sample_size
        """
        generated_docs = []

        for t in range(0, self.sample_size):
            new_instance_explained = copy.copy(instance_explained)

            for sel_feat in range(2, qid_data.shape[1]):
                sigma = np.std(self.data.iloc[:, sel_feat].values)
                z = np.random.normal(0, 1)
                new_instance_explained[sel_feat] = z * sigma + instance_explained[sel_feat]

            generated_docs.append(new_instance_explained)

        return generated_docs

    def dlime_sampling(self, instance_explained, qid_data):
        """Generates a batch of syntethic samples implementing dlime sampling

        Args:
            instance_explained: document in qid_data to explain
            qid_data: pandas series corresponding to
            the documents returned by a specific query
        Returns:
            A list of documents (pandas series) of length
            sample_size
        """
        generated_docs = self.latin_hypercube_sampling(instance_explained, qid_data)
        cleaned_docs = [series[2:] for series in generated_docs] #without qid and label
        clustering = AgglomerativeClustering(num_clust(cleaned_docs)).fit(cleaned_docs)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cleaned_docs)

        instance = instance_explained.values[2:].astype(np.float32)
        distance, indices = nbrs.kneighbors(instance.reshape(1, -1))
        indices = indices.flatten()

        p_label = clustering.labels_[indices][0]

        compar = p_label == clustering.labels_
        subset = [doc for doc, flag in zip(generated_docs, compar) if flag]
        return subset

class Explanations:
    """Generates explanations for a document within a ranked set of documents.
    The explanations are scores assigned to each document's feature indicating
    its relevance in the ranking's output. Two types of explanations are
    generated: those from the orginial interpretable model and those from the
    LIME method.
    """

    def __init__(self, num_features):
        """Init.

        Args:
            num_features: number of features (integer)
        """
        self.kernel_width = np.sqrt(num_features) * .75
        self.ratio_zeros = None

    def subscores_GAM(self, instances, idx):
        """Explains the position of a document in the ranking
        using the interpretable ranking model

        Args:
            instances: pandas series of documents returned by a specific query
            idx: index (integer) of the document to explain in instances
        Returns:
            A list of subscores (float) of length number_features
        """
        acum_subscores = []
        tensors = tf.convert_to_tensor(instances)
        for fea in _name_features:
            tf_predictor = _loaded_model.signatures[fea + '_subscore']
            subscores = tf_predictor(tensors)
            acum_subscores.append(subscores['outputs'][idx])
        return tf.stack(acum_subscores)

    def predict_GAM(self, instances):
        """Predicts the documents' ranking scores using the ranking model

        Args:
            instances: pandas series of documents returned by a specific query
            idx: index (integer) of the document to explain in instances
        Returns:
            A list of scores (float) of equal length as instances
        """
        tf_example_predictor = _loaded_model.signatures['predict']
        scores = tf_example_predictor(tf.convert_to_tensor(instances))['output']
        return scores

    def kernel(self, dist):
        """Calculates the similarity kernel

        Args:
            dist: euclidean distances
        Returns:
            Weights in (0,1)
        """
        return np.sqrt(np.exp(-(dist ** 2) / self.kernel_width ** 2))
    def top_k_binary(self, ranked_all, doc_idx, k):
        """Labels the samples (0,1) depending on their position
        in the ranking

        Args:
            ranked_all: list of lists (rankings) of integers from 0 to number of documents-1, indicating
            the position of each document in the predicted ranking.
            doc_idx: index of the document to be explained
            k: threshold, lower ranked documents are labeled as zero
        Returns:
            List of binary labels (integer) of length ranked_all
        """
        labels = []
        for ranked in ranked_all:
            if ranked[doc_idx] < k:
                labels.append(1)
            else:
                labels.append(0)
        return labels
    def get_explanations(self, qid_data, position_rank, sampling):
        """Explains the position of a document in the predicted
        ranking

        Args:
            qid_data: pandas series of documents returned by a specific query. Shape (n_documents, n_features + qid + label)
            position_rank: ranking position of the document to be explained (integer)
            sampling: sampling_generator object
        Returns:
           Two ndarrays of subscores, corresponging to the explanations: the first
           are the explantions from the interpretable ranking model; the second
           are the explanations from LIME
        """
        docs = serialize_all(qid_data)

        original_scores = self.predict_GAM(docs)
        base_rank = rankdata([-1 * i for i in original_scores]).astype(int) - 1
        doc_idx = np.argmax(
            base_rank == position_rank - 1)  # select the index of the instance that falls in position_rank
        gam_explanations = self.subscores_GAM(docs, doc_idx)

        instance_explained = copy.copy(qid_data.iloc[doc_idx])

        generated_docs = sampling(instance_explained, qid_data)

        generated_predictions = []

        # Replaces the instance for the new sample, and ranks again using GAM
        for t in range(0, len(generated_docs)):
            temp_docs = copy.copy(docs)
            temp_docs[doc_idx] = serialize_all(generated_docs[t].to_frame().T)[0]
            genere_pred = self.predict_GAM(temp_docs)
            generated_predictions.append(genere_pred)
        ranked_all = []

        # Turns the predicition scores to a ranked list of documents
        for gen_pred in generated_predictions:
            ranked_all.append(rankdata([-1 * i for i in gen_pred]).astype(int) - 1)
        ranked_all = np.array(ranked_all)

        # Labels samples
        labels = self.top_k_binary(ranked_all, doc_idx, position_rank)
        self.ratio_zeros = labels.count(0) / len(labels)

        # SMOTE for class imbalance
        smote = SMOTE()

        samples_fea = [series[2:] for series in generated_docs]
        X_resampled, y_resampled = smote.fit_resample(np.array(samples_fea), labels)

        gen_docs = []
        for i in range(0, len(generated_docs)):
            gen_docs.append(generated_docs[i].values[2:])
        gen_docs = np.array(gen_docs).astype(np.float32)

        # Calculates weights for each sample depending on their distance to instance
        i_explained = instance_explained.values[2:].astype(np.float32)
        distances = pairwise_distances(
            X_resampled, i_explained.reshape(1, -1), metric='euclidean'
        ).ravel()
        k_weights = self.kernel(distances).astype(np.float32)
        clf = RidgeClassifier().fit(X_resampled, y_resampled, sample_weight=k_weights)
        lime_explanations = tf.transpose(clf.coef_)

        return gam_explanations, lime_explanations

def iter_overlap(list1, list2):
    """Calculates the % of overlap for two equal sized lists of ranked values in incremental sets
    of five, from 0 to number features

    Args:
        list1: list of ranked values (integers)
        list2: list of ranked values (integers). Same length as list1
    Returns:
       List of floats
    """
    list_overlap = []
    for x in range(5, len(_name_features) + 5, 5):
        included_values = [i for i in range(x)]
        indices1 = np.where(np.isin(list1, included_values))[0]
        indices2 = np.where(np.isin(list2, included_values))[0]
        inter = np.size(np.intersect1d(indices1, indices2))
        list_overlap.append(inter / x)
    return list_overlap
def rbo(list1, list2, p=0.9):
    """Calculates the rbo score for two equal sized ranks

    Args:
        list1: list of ranked values (integers)
        list2: list of ranked values (integers). Same length as list1
        p: weight assigned to items at different positions in the ranked lists, ranges [0,1]
    Returns:
       rbo (float)
    """
    # tail recursive helper function
    def helper(ret, i, d):
        l1 = set(list1[:i]) if i < len(list1) else set(list1)
        l2 = set(list2[:i]) if i < len(list2) else set(list2)
        a_d = len(l1.intersection(l2)) / i
        term = math.pow(p, i) * a_d
        if d == i:
            return ret + term
        return helper(ret + term, i + 1, d)

    k = max(len(list1), len(list2))
    x_k = len(set(list1).intersection(set(list2)))
    summation = helper(0, 1, k)

    return ((float(x_k) / k) * math.pow(p, k)) + ((1 - p) / p * summation)


def calculate_metrics(gam_explanations, lime_explanations, k_top):
    """Calculates the rbo and the overlap metrics for two set of explanations

    Args:
        gam_explanations: ndarray of subscores from GAM model
        lime_explanations: ndarray of subscores from LIME
    Returns:
       rbo (float) and overlap list (floats)
    """
    flatten_GAM = gam_explanations.numpy().flatten()
    flatten_LIME = lime_explanations.numpy().flatten()

    ranked_GAM = rankdata([-1 * i for i in flatten_GAM]).astype(int) - 1
    ranked_LIME = rankdata([-1 * i for i in flatten_LIME]).astype(int) - 1
    rbo_metric = rbo(ranked_GAM, ranked_LIME, k_top)
    overlap_metric = iter_overlap(ranked_GAM, ranked_LIME)
    return rbo_metric, overlap_metric


def main():
    tf.compat.v1.set_random_seed(1234)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    df_test = pd.read_csv(_test_yahoo)
    grouped_qid = df_test.groupby('qid')
    qids = [28587, 23068, 28700, 24240, 23890, 28483, 23132, 29379, 23022, 24923, 25000, 29889, 25191, 25207, 25226,
            28035, 28203, 28670, 28286, 28276]
    ranked_pos = 10
    test = []
    start = time.time()
    print('Explanations in progress')
    for strategy in ['smote']:
        sampling = Sampling_generator(df_test, sample_size=1000, strategy_name=strategy)
        qids_dict = {'imbalance': [], 'rbo': [], 'overlap': []}
        for key in qids:
            grouped_data = grouped_qid.get_group(key)
            exp = Explanations(len(_name_features))
            gam_exp, lime_exp = exp.get_explanations(grouped_data, ranked_pos, sampling.technique)
            print(exp.ratio_zeros)
            rbo_m, overlap_m = calculate_metrics(gam_exp, lime_exp, 0.95)
            qids_dict['imbalance'].append(exp.ratio_zeros)
            qids_dict['rbo'].append(rbo_m)
            qids_dict['overlap'].append(overlap_m)
            print(rbo_m)
            print('Finished ' + strategy + ' with qid: ' + str(key))
        test.append(qids_dict)
    end = time.time()
    print('Time took for explanations: {} '.format(end - start))

    print('*' * 24 + ' RESULTS ' + '*' * 24)
    print('=' * 16 + ' ' + str(ranked_pos) + 'th ranked doc ' + '=' * 16)
    print(test)
    print('=' * 64)


if __name__ == "__main__":
    main()