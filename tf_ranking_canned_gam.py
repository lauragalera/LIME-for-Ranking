# Copyright 2022 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications by Laura Galera

"""TF-Ranking example code for training a canned GAM estimator.

The supported proto formats are listed at ../python/data.py.
--------------------------------------------------------------------------------
Sample command lines:

MODEL_DIR=/tmp/output && \
TRAIN=tensorflow_ranking/examples/data/train_numerical_elwc.tfrecord && \
EVAL=tensorflow_ranking/examples/data/vali_numerical_elwc.tfrecord && \
rm -rf $MODEL_DIR && \
bazel build -c opt \
tensorflow_ranking/examples/tf_ranking_canned_gam_py_binary && \
./bazel-bin/tensorflow_ranking/examples/tf_ranking_canned_gam_py_binary \
--train_input_pattern=$TRAIN \
--eval_input_pattern=$EVAL \
--model_dir=$MODEL_DIR

You can use TensorBoard to display the training results stored in $MODEL_DIR.

Notes:
  * Use --alsologtostderr if the output is not printed into screen.
"""

from absl import flags

import tensorflow as tf
import tensorflow_ranking as tfr


flags.DEFINE_string("train_input_pattern", None,
                    "Input file path used for training.")
flags.DEFINE_string("eval_input_pattern", None,
                    "Input file path used for eval.")
flags.DEFINE_string("model_dir", None, "Output directory for models.")
flags.DEFINE_integer("batch_size", 32, "The batch size for train.")
flags.DEFINE_integer("num_train_steps", 2000, "Number of steps for train.")
flags.DEFINE_integer("num_eval_steps", 10, "Number of steps for evaluation.")
flags.DEFINE_integer("checkpoint_secs", 30,
                     "Saves a model checkpoint every checkpoint_secs seconds.")
flags.DEFINE_integer("num_checkpoints", 100,
                     "Saves at most num_checkpoints checkpoints in workspace.")

flags.DEFINE_integer("num_features", 96, "Number of features per example.")
flags.DEFINE_integer(
    "list_size", 50,
    "List size used for training. Use None for dynamic list size.")

flags.DEFINE_float("learning_rate", 0.05, "Learning rate for optimizer.")
flags.DEFINE_float("dropout", 0.5, "The dropout rate before output layer.")
flags.DEFINE_list("hidden_layer_dims", ["16", "8"],
                  "Sizes for hidden layers.")
flags.DEFINE_string("loss", "approx_ndcg_loss",
                    "The RankingLossKey for the loss function.")
flags.DEFINE_bool("convert_labels_to_binary", False,
                  "If true, relevance labels are set to either 0 or 1.")
flags.DEFINE_bool("listwise_inference", False,
                  "If true, exports ELWC while serving.")

FLAGS = flags.FLAGS

_LABEL_FEATURE = "relevance_label"


def example_feature_columns():
  """Returns the example feature columns."""
  feature_names = ['covered_query_term_number_body','covered_query_term_number_anchor','covered_query_term_number_title',
                 'covered_query_term_number_url','covered_query_term_number_whole_document','covered_query_term_ratio_body',
                'covered_query_term_ratio_anchor','covered_query_term_ratio_title','covered_query_term_ratio_url',
                 'covered_query_term_ratio_whole_document', 'stream_length_body', 'stream_length_anchor',
                'stream_length_title','stream_length_url','stream_length_whole_document','sum_term_freq_body','sum_term_freq_anchor','sum_term_freq_title',
                 'sum_term_freq_url','sum_term_freq_whole_document','min_term_freq_body','min_term_freq_anchor','min_term_freq_title',
                 'min_term_freq_url','min_term_freq_whole_document','max_term_freq_body','max_term_freq_anchor','max_term_freq_title',
                 'max_term_freq_url','max_term_freq_whole_document','mean_term_freq_body','mean_term_freq_anchor','mean_term_freq_title',
                 'mean_term_freq_url','mean_term_freq_whole_document','sum_stream_length_normalized_term_freq_body','sum_stream_length_normalized_term_freq_anchor',
                 'sum_stream_length_normalized_term_freq_title','sum_stream_length_normalized_term_freq_url','sum_stream_length_normalized_term_whole_document',
                 'min_stream_length_normalized_term_freq_body','min_stream_length_normalized_term_freq_anchor','min_stream_length_normalized_term_freq_title',
                 'min_stream_length_normalized_term_freq_url','min_stream_length_normalized_term_freq_whole_document','max_stream_length_normalized_term_freq_body',
                 'max_stream_length_normalized_term_freq_anchor','max_stream_length_normalized_term_freq_title','max_stream_length_normalized_term_freq_url',
                 'max_stream_length_normalized_term_freq_whole_document','mean_stream_length_normalized_term_freq_body','mean_stream_length_normalized_term_freq_anchor',
                 'mean_stream_length_normalized_term_freq_title','mean_stream_length_normalized_term_freq_url','mean_stream_length_normalized_term_freq_whole_document','boolean_model_body',
                 'boolean_model_anchor','boolean_model_title','boolean_model_url','boolean_model_whole_document','vector_space_model_body',
                 'vector_space_model_anchor','vector_space_model_title','vector_space_model_url','vector_space_model_whole_document','BM25_body',
                 'BM25_anchor','BM25_title','BM25_url','BM25_whole_document','LMIR.ABS_body','LMIR.ABS_anchor','LMIR.ABS_title','LMIR.ABS_url',
                 'LMIR.ABS_whole_document','LMIR.DIR_body','LMIR.DIR_anchor','LMIR.DIR_title','LMIR.DIR_url','LMIR.DIR_whole_document','LMIR.JM_body',
                 'LMIR.JM_anchor','LMIR.JM_title','LMIR.JM_url','LMIR.JM_whole_document','num_slash_url','length_url','inlink_number','outlink_number',
                 'pagerank','siterank','qualityscore','qualityscore2','query_url_click_count','url_click_count','url_dwell_time']
  return {
      name:
      tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
      for name in feature_names
  }


def train_and_eval():
  """Train and Evaluate."""
  optimizer = tf.compat.v1.train.AdagradOptimizer(
      learning_rate=FLAGS.learning_rate)

  estimator = tfr.estimator.make_gam_ranking_estimator(
      example_feature_columns(),
      FLAGS.hidden_layer_dims,
      optimizer=optimizer,
      learning_rate=FLAGS.learning_rate,
      loss=FLAGS.loss,
      loss_reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE,
      activation_fn=tf.nn.relu,
      dropout=FLAGS.dropout,
      use_batch_norm=True,
      model_dir=FLAGS.model_dir)

  hparams = {"train_input_pattern": FLAGS.train_input_pattern,
             "eval_input_pattern": FLAGS.eval_input_pattern,
             "learning_rate": FLAGS.learning_rate,
             "train_batch_size": FLAGS.batch_size,
             "eval_batch_size": FLAGS.batch_size,
             "predict_batch_size": FLAGS.batch_size,
             "num_train_steps": FLAGS.num_train_steps,
             "num_eval_steps": FLAGS.num_eval_steps,
             "checkpoint_secs": FLAGS.checkpoint_secs,
             "num_checkpoints": FLAGS.num_checkpoints,
             "loss": FLAGS.loss,
             "list_size": FLAGS.list_size,
             "convert_labels_to_binary": FLAGS.convert_labels_to_binary,
             "listwise_inference": FLAGS.listwise_inference,
             "model_dir": FLAGS.model_dir}

  ranking_pipeline = tfr.ext.pipeline.RankingPipeline(
      {},
      example_feature_columns(),
      hparams,
      estimator=estimator,
      label_feature_name=_LABEL_FEATURE,
      label_feature_type=tf.int64)

  ranking_pipeline.train_and_eval()


def main(_):
  tf.compat.v1.set_random_seed(1234)
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  train_and_eval()


if __name__ == "__main__":
  flags.mark_flag_as_required("train_input_pattern")
  flags.mark_flag_as_required("eval_input_pattern")
  flags.mark_flag_as_required("model_dir")

  tf.compat.v1.app.run()
