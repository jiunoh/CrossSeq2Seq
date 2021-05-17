# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Download and preprocess WMT17 ende training and evaluation datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import tarfile

# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
from absl import logging
import six
from six.moves import range
from six.moves import urllib
from six.moves import zip
import tensorflow.compat.v1 as tf

from official.utils.flags import core as flags_core
from official.nlp.transformer.utils.tokenizer import EOS_ID
import official.nlp.transformer.utils.tokenizer as tokenizer

import random
# pylint: enable=g-bad-import-order

# Data sources for training/evaluating the transformer translation model.
# If any of the training sources are changed, then either:
#   1) use the flag `--search` to find the best min count or
#   2) update the _TRAIN_DATA_MIN_COUNT constant.
# min_count is the minimum number of times a token must appear in the data
# before it is added to the vocabulary. "Best min count" refers to the value
# that generates a vocabulary set that is closest in size to _TARGET_VOCAB_SIZE.


# Use pre-defined minimum count to generate subtoken vocabulary.
_TRAIN_DATA_MIN_COUNT = 6

# Vocabulary constants
_TARGET_VOCAB_SIZE = 32768  # Number of subtokens in the vocabulary list.
_TARGET_THRESHOLD = 327  # Accept vocabulary if size is within this threshold
VOCAB_FILE = "vocab.en_wmt14.10000"

# Strings to inclue in the generated files.
_PREFIX = "pretrain_EN"
_TRAIN_TAG = "train"
_EVAL_TAG = "dev"  # Following WMT and Tensor2Tensor conventions, in which the
# evaluation datasets are tagged as "dev" for development.

# Number of files to split train and evaluation data
_TRAIN_SHARDS = 100
_EVAL_SHARDS = 1

MASK_PROB = 0.25
MAX_PRED_PER_SEQ = 20
rng = random.Random(12345)


def txt_line_iterator(path):
  """Iterate through lines of file."""
  with tf.io.gfile.GFile(path) as f:
    for line in f:
      yield line.strip()


def write_file(writer, filename):
  """Write all of lines from file using the writer."""
  for line in txt_line_iterator(filename):
    writer.write(line)
    writer.write("\n")


###############################################################################
# Data preprocessing
###############################################################################
def drop_sequence(subtokenizer, token_ids):
    cand_indexes = []
    is_subword = False
    for (i, token_id) in enumerate(token_ids):
      if token_id != EOS_ID:
        if not subtokenizer.subtoken_list[token_id].endswith('_'):
            if is_subword:
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
                is_subword = True
        else:
            if is_subword:
                cand_indexes[-1].append(i)
                is_subword = False
            else:
                cand_indexes.append([i])

    rng.shuffle(cand_indexes)
    tokens_after_dropping = token_ids[:]
    num_to_predict = min(MAX_PRED_PER_SEQ, max(1, int(round(len(token_ids) * MASK_PROB))))
    dropped_tokens = []
    covered_indexes = set()

    tokens_dropped = [0 for i in token_ids]
    for index_set in cand_indexes:
        if len(dropped_tokens) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(dropped_tokens) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            tokens_after_dropping[index] = -1
            dropped_tokens.append(index)
            tokens_dropped[index] = token_ids[index]

    inputs = []
    for token_id in tokens_after_dropping:
        if token_id != -1:
            inputs.append(token_id)

    targets = []
    for token_id in tokens_dropped:
        if token_id != 0:
            targets.append(token_id)
    targets.append(EOS_ID)

    assert len(dropped_tokens) <= num_to_predict

    return inputs, targets


def encode_and_save_files(
    subtokenizer, data_dir, raw_files, tag, total_shards):
  """Save data from files as encoded Examples in TFrecord format.

  Args:
    subtokenizer: tokenizer object that will be used to encode the strings.
    data_dir: The directory in which to write the examples
    raw_files: A tuple of (input, target) data files. Each line in the input and
      the corresponding line in target file will be saved in a tf.Example.
    tag: String that will be added onto the file names.
    total_shards: Number of files to divide the data into.

  Returns:
    List of all files produced.
  """
  # Create a file for each shard.
  filepaths = [shard_filename(data_dir, tag, n + 1, total_shards)
               for n in range(total_shards)]

  if all_exist(filepaths):
    logging.info("Files with tag %s already exist." % tag)
    return filepaths

  logging.info("Saving files with tag %s." % tag)
  input_file = raw_files[0]
  target_file = raw_files[1]

  # Write examples to each shard in round robin order.
  tmp_filepaths = [six.ensure_str(fname) + ".incomplete" for fname in filepaths]
  writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_filepaths]
  counter, shard = 0, 0
  for counter, (input_line, target_line) in enumerate(zip(
      txt_line_iterator(input_file), txt_line_iterator(target_file))):
    if counter > 0 and counter % 100000 == 0:
      logging.info("\tSaving case %d." % counter)
    
    ids = subtokenizer.encode(input_line, add_eos=True)
    inputs, targets = drop_sequence(subtokenizer, ids)
    example = dict_to_example({"inputs": inputs, "targets": targets})

    writers[shard].write(example.SerializeToString())
    shard = (shard + 1) % total_shards
  for writer in writers:
    writer.close()

  for tmp_name, final_name in zip(tmp_filepaths, filepaths):
    tf.gfile.Rename(tmp_name, final_name)

  logging.info("Saved %d Examples", counter + 1)
  return filepaths


def shard_filename(path, tag, shard_num, total_shards):
  """Create filename for data shard."""
  return os.path.join(
      path, "%s-%s-%.5d-of-%.5d" % (_PREFIX, tag, shard_num, total_shards))


def shuffle_records(fname):
  """Shuffle records in a single file."""
  logging.info("Shuffling records in file %s" % fname)

  # Rename file prior to shuffling
  tmp_fname = six.ensure_str(fname) + ".unshuffled"
  tf.gfile.Rename(fname, tmp_fname)

  reader = tf.io.tf_record_iterator(tmp_fname)
  records = []
  for record in reader:
    records.append(record)
    if len(records) % 100000 == 0:
      logging.info("\tRead: %d", len(records))

  random.shuffle(records)

  # Write shuffled records to original file name
  with tf.python_io.TFRecordWriter(fname) as w:
    for count, record in enumerate(records):
      w.write(record)
      if count > 0 and count % 100000 == 0:
        logging.info("\tWriting record: %d" % count)

  tf.gfile.Remove(tmp_fname)


def dict_to_example(dictionary):
  """Converts a dictionary of string->int to a tf.Example."""
  features = {}
  for k, v in six.iteritems(dictionary):
    features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
  return tf.train.Example(features=tf.train.Features(feature=features))


def all_exist(filepaths):
  """Returns true if all files in the list exist."""
  for fname in filepaths:
    if not tf.gfile.Exists(fname):
      return False
  return True


def make_dir(path):
  if not tf.gfile.Exists(path):
    logging.info("Creating directory %s" % path)
    tf.gfile.MakeDirs(path)


def main(unused_argv):
  """Obtain training and evaluation data for the Transformer model."""

  # Create tokenizer based on the training files.
  logging.info("Step 1: Loading tokenizer")
  train_en = FLAGS.data_dir+'/EN_TRAIN_CORPUS_NAME'
  val_en = FLAGS.data_dir+'/EN_VAL_CORPUS_NAME'

  vocab_file = os.path.join(FLAGS.data_dir, VOCAB_FILE)
  subtokenizer = tokenizer.init_from_files(
      vocab_file, [train_en], _TARGET_VOCAB_SIZE, _TARGET_THRESHOLD,
      min_count=None if FLAGS.search else _TRAIN_DATA_MIN_COUNT)

  compiled_train_files = (train_en, train_en)
  compiled_eval_files = (val_en, val_en)

  # Tokenize and save data as Examples in the TFRecord format.
  logging.info("Step 3: Preprocessing and saving data")
  train_tfrecord_files = encode_and_save_files(
      subtokenizer, FLAGS.data_dir, compiled_train_files, _TRAIN_TAG,
      _TRAIN_SHARDS)
  encode_and_save_files(
      subtokenizer, FLAGS.data_dir, compiled_eval_files, _EVAL_TAG,
      _EVAL_SHARDS)

  for fname in train_tfrecord_files:
    shuffle_records(fname)


def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", short_name="dd", default="/tmp/translate_ende",
      help=flags_core.help_wrap(
          "Directory for where the translate_ende_wmt32k dataset is saved."))
  flags.DEFINE_string(
      name="raw_dir", short_name="rd", default="/tmp/translate_ende_raw",
      help=flags_core.help_wrap(
          "Path where the raw data will be downloaded and extracted."))
  flags.DEFINE_bool(
      name="search", default=False,
      help=flags_core.help_wrap(
          "If set, use binary search to find the vocabulary set with size"
          "closest to the target size (%d)." % _TARGET_VOCAB_SIZE))


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  define_data_download_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
