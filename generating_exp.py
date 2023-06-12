# Copyright 2022 The TensorFlow Ranking Authors.
#
# Author: Laura Galera Alfaro

r"""Example code for generating ranking explanations
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

_model_yahoo = "master/TFM/output_yahoo/export/latest_model/1684750780"
_model_mslr = "master/TFM/output_mslr/export/latest_model/1684752279"
_test_yahoo = "master/TFM/datasets/yahoo/test_yahoo.csv"
_test_mslr = "master/TFM/datasets/MSLR-WEB10K/test_mslr.csv"

_name_features = [str(i + 1) for i in range(0, 100)]
_name_features_mslr = ['covered_query_term_number_body', 'covered_query_term_number_anchor',
                       'covered_query_term_number_title',
                       'covered_query_term_number_url', 'covered_query_term_number_whole_document',
                       'covered_query_term_ratio_body',
                       'covered_query_term_ratio_anchor', 'covered_query_term_ratio_title',
                       'covered_query_term_ratio_url',
                       'covered_query_term_ratio_whole_document', 'stream_length_body', 'stream_length_anchor',
                       'stream_length_title', 'stream_length_url', 'stream_length_whole_document', 'sum_term_freq_body',
                       'sum_term_freq_anchor', 'sum_term_freq_title',
                       'sum_term_freq_url', 'sum_term_freq_whole_document', 'min_term_freq_body',
                       'min_term_freq_anchor', 'min_term_freq_title',
                       'min_term_freq_url', 'min_term_freq_whole_document', 'max_term_freq_body',
                       'max_term_freq_anchor', 'max_term_freq_title',
                       'max_term_freq_url', 'max_term_freq_whole_document', 'mean_term_freq_body',
                       'mean_term_freq_anchor', 'mean_term_freq_title',
                       'mean_term_freq_url', 'mean_term_freq_whole_document',
                       'sum_stream_length_normalized_term_freq_body', 'sum_stream_length_normalized_term_freq_anchor',
                       'sum_stream_length_normalized_term_freq_title', 'sum_stream_length_normalized_term_freq_url',
                       'sum_stream_length_normalized_term_whole_document',
                       'min_stream_length_normalized_term_freq_body', 'min_stream_length_normalized_term_freq_anchor',
                       'min_stream_length_normalized_term_freq_title',
                       'min_stream_length_normalized_term_freq_url',
                       'min_stream_length_normalized_term_freq_whole_document',
                       'max_stream_length_normalized_term_freq_body',
                       'max_stream_length_normalized_term_freq_anchor', 'max_stream_length_normalized_term_freq_title',
                       'max_stream_length_normalized_term_freq_url',
                       'max_stream_length_normalized_term_freq_whole_document',
                       'mean_stream_length_normalized_term_freq_body', 'mean_stream_length_normalized_term_freq_anchor',
                       'mean_stream_length_normalized_term_freq_title', 'mean_stream_length_normalized_term_freq_url',
                       'mean_stream_length_normalized_term_freq_whole_document', 'boolean_model_body',
                       'boolean_model_anchor', 'boolean_model_title', 'boolean_model_url',
                       'boolean_model_whole_document', 'vector_space_model_body',
                       'vector_space_model_anchor', 'vector_space_model_title', 'vector_space_model_url',
                       'vector_space_model_whole_document', 'BM25_body',
                       'BM25_anchor', 'BM25_title', 'BM25_url', 'BM25_whole_document', 'LMIR.ABS_body',
                       'LMIR.ABS_anchor', 'LMIR.ABS_title', 'LMIR.ABS_url',
                       'LMIR.ABS_whole_document', 'LMIR.DIR_body', 'LMIR.DIR_anchor', 'LMIR.DIR_title', 'LMIR.DIR_url',
                       'LMIR.DIR_whole_document', 'LMIR.JM_body',
                       'LMIR.JM_anchor', 'LMIR.JM_title', 'LMIR.JM_url', 'LMIR.JM_whole_document', 'num_slash_url',
                       'length_url', 'inlink_number', 'outlink_number',
                       'pagerank', 'siterank', 'qualityscore', 'qualityscore2', 'query_url_click_count',
                       'url_click_count', 'url_dwell_time']

_loaded_model = tf.saved_model.load(_model_yahoo)


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_all(examples):
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


def top_k_binary(ranked_all, doc_idx, k=10):
    labels = []
    for ranked in ranked_all:
        if ranked[doc_idx] < k:
            labels.append(1)
        else:
            labels.append(0)
    return labels


class Explanations:
    def __init__(self, data, sample_size=20):
        self.data = data
        self.sample_size = sample_size
        self.kernel_width = np.sqrt(data.shape[1]) * .75
        self.ratio_zeros = None

    def subscores_GAM(self, instances, idx):
        acum_subscores = []
        tensors = tf.convert_to_tensor(instances)
        for fea in _name_features:
            tf_predictor = _loaded_model.signatures[fea + '_subscore']
            subscores = tf_predictor(tensors)
            acum_subscores.append(subscores['outputs'][idx])
        return tf.stack(acum_subscores)

    def predict_GAM(self, instances):
        tf_example_predictor = _loaded_model.signatures['predict']
        scores = tf_example_predictor(tf.convert_to_tensor(instances))['output']
        return scores

    def kernel(self, d):
        # similarity or weight based on the Gaussian kernel function
        return np.sqrt(np.exp(-(d ** 2) / self.kernel_width ** 2))

    def empirical_sampling(self, instance_explained, qid_data):
        generated_docs = []

        for t in range(0, self.sample_size):
            new_instance_explained = copy.copy(instance_explained)

            for sel_feat in range(2, qid_data.shape[1]):
                new_instance_explained[sel_feat] = np.random.choice(qid_data.iloc[:, sel_feat], 1)[0]

            generated_docs.append(new_instance_explained)
        return generated_docs

    def uniformed_sampling(self, k, instance_explained, qid_data):
        generated_docs = []

        nbrs = NearestNeighbors(n_neighbors = k + 1)
        nbrs.fit(qid_data.iloc[:, 2:])
        instance = instance_explained.values[2:].astype(np.float32)
        distance, indices = nbrs.kneighbors(instance.reshape(1, -1))
        indices = indices.flatten()[1:]
        for t in range (0, self.sample_size):
            new_instance_explained = copy.copy(instance_explained)
            indx_neighbor = np.random.choice(indices)
            random_neighbor = qid_data.iloc[indx_neighbor]
            for sel_feat in range(2, qid_data.shape[1]):
                min_val = np.min([random_neighbor.iloc[sel_feat], instance_explained.iloc[sel_feat]])
                max_val = np.max([random_neighbor.iloc[sel_feat], instance_explained.iloc[sel_feat]])
                perturbation_range = (min_val, max_val)
                perturbation = np.random.uniform(*perturbation_range)
                new_instance_explained[sel_feat] = perturbation
            generated_docs.append(new_instance_explained)
        return generated_docs

    def gaussian_inverse_sampling(self, instance_explained, qid_data):
        generated_docs = []

        for t in range(0, self.sample_size):
            new_instance_explained = copy.copy(instance_explained)

            for sel_feat in range(2, qid_data.shape[1]):
                sigma = np.std(qid_data.iloc[:, sel_feat].values)
                z = np.random.normal(0, 1)
                new_instance_explained[sel_feat] = z * sigma + instance_explained[sel_feat]

            generated_docs.append(new_instance_explained)
        return generated_docs

    def get_explanations(self, qid_data, position_rank, k_binary, sampling):

        docs = serialize_all(qid_data)

        original_scores = self.predict_GAM(docs)
        base_rank = rankdata([-1 * i for i in original_scores]).astype(int) - 1
        doc_idx = np.argmax(
            base_rank == position_rank - 1)  # select the index of the instance that falls in position_rank
        gam_explanations = self.subscores_GAM(docs, doc_idx)

        instance_explained = copy.copy(qid_data.iloc[doc_idx])

        # Returns list of new documents samples
        if sampling == 'empirical':
            generated_docs = self.empirical_sampling(instance_explained, qid_data)
        elif sampling == 'uniformed':
            generated_docs = self.uniformed_sampling(10, instance_explained, qid_data)
        elif sampling == 'gaussian':
            generated_docs = self.gaussian_inverse_sampling(instance_explained, qid_data)

        generated_predictions = []

        # Replace the instance for the new sample, and ranks again using GAM
        for t in range(0, self.sample_size):
            temp_docs = copy.copy(docs)
            temp_docs[doc_idx] = serialize_all(generated_docs[t].to_frame().T)[0]
            genere_pred = self.predict_GAM(temp_docs)
            generated_predictions.append(genere_pred)
        ranked_all = []

        # Turns the predicition scores to a ranked list of documents
        for gen_pred in generated_predictions:
            ranked_all.append(rankdata([-1 * i for i in gen_pred]).astype(int) - 1)
        ranked_all = np.array(ranked_all)

        # Label samples
        labels = top_k_binary(ranked_all, doc_idx, k_binary)
        self.ratio_zeros = labels.count(0) / len(labels)

        # SMOTE for class imbalance
        smote = SMOTE()

        samples_fea = [series[2:] for series in generated_docs]
        X_resampled, y_resampled = smote.fit_resample(np.array(samples_fea), labels)

        gen_docs = []
        for i in range(0, len(generated_docs)):
            gen_docs.append(generated_docs[i].values[2:])
        gen_docs = np.array(gen_docs).astype(np.float32)

        # Calculates weight for each sample depending on distance to instance
        i_explained = instance_explained.values[2:].astype(np.float32)
        distances = np.linalg.norm(X_resampled - i_explained, axis=1)
        k_weights = self.kernel(distances).astype(np.float32)
        clf = RidgeClassifier().fit(X_resampled, y_resampled, sample_weight=k_weights)
        lime_explanations = tf.transpose(clf.coef_)

        return gam_explanations, lime_explanations


def rbo(list1, list2, p=0.9):
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
    flatten_GAM = gam_explanations.numpy().flatten()
    flatten_LIME = lime_explanations.numpy().flatten()

    ranked_GAM = rankdata([-1 * i for i in flatten_GAM]).astype(int) - 1
    ranked_LIME = rankdata([-1 * i for i in flatten_LIME]).astype(int) - 1

    return rbo(ranked_GAM, ranked_LIME, k_top)


def main():
    tf.compat.v1.set_random_seed(1234)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    df_test = pd.read_csv(_test_yahoo)
    grouped_qid = df_test.groupby('qid')
    qids = [23843, 23875, 23996, 24217, 24251, 24626, 24633, 24709, 24712, 24726, 24746, 24754, 24762, 24832, 24840, 24848, 24869, 24878, 24883, 24918, 24921, 24922, 24961, 25017, 25023, 25051, 25054, 25086, 25088, 25133, 25182, 25215, 25224, 27890, 27987, 28001, 28024, 28026, 28040, 28041, 28042, 28049, 28052, 28073, 28106, 28130, 28150, 28153, 28175, 28183, 28185, 28190, 28269, 28274, 28289, 28310, 28375, 28382, 28397, 28398, 28412, 28417, 28420, 28462, 28472, 28482, 28544, 28550, 28552, 28557, 28577, 28596, 28615, 28654, 28657, 28659, 28694, 28715, 28757, 28805, 28836, 28842, 28845, 28866, 28890, 28969, 28995, 29029, 29050, 29113, 29190, 29213, 29249, 29256, 29334, 29357, 29404, 29438, 29462, 29487, 29625, 29681, 29725, 29760, 29761, 29768, 29776, 29781, 29787, 29790, 29823, 29826, 29863, 29885, 29895]
    ranked_pos = 10
    test = []
    start = time.time()
    print('Start making explanations')
    for strategy in ['gaussian']:
        qids_dict = {'imbalance': [], 'rbo': []}
        for key in qids:
            group_data = grouped_qid.get_group(key)
            lime = Explanations(df_test, sample_size=1000)
            gam_exp, lime_exp = lime.get_explanations(group_data, ranked_pos, ranked_pos, strategy)
            print(lime.ratio_zeros)
            metric = calculate_metrics(gam_exp, lime_exp, 0.95)
            qids_dict['imbalance'].append(lime.ratio_zeros)
            qids_dict['rbo'].append(metric)
            print(metric)
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
