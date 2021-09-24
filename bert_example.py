# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Build BERT Examples from text (source, target) pairs."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import numpy as np
from bert import tokenization
import tagging
import tagging_converter
import tensorflow as tf
import copy
from typing import Mapping, MutableSequence, Optional, Sequence, Text


class BertExample(object):
  """Class for training and inference examples for BERT.

  Attributes:
    editing_task: The EditingTask from which this example was created. Needed
      when realizing labels predicted for this example.
    features: Feature dictionary.
  """

  def __init__(self, input_ids,
               input_mask,
               segment_ids, labels,
               labels_mask,
               token_start_indices,
               task, default_label, add_mask, add_index, dec_inputs, dec_targets, nums_add):
    input_len = len(input_ids)
    if not (input_len == len(input_mask) and input_len == len(segment_ids) and
            input_len == len(labels) and input_len == len(labels_mask)):
      raise ValueError(
          'All feature lists should have the same length ({})'.format(
              input_len))

    self.features = collections.OrderedDict([
        ('input_ids', input_ids),
        ('input_mask', input_mask),
        ('segment_ids', segment_ids),
        ('labels', labels),
        ('labels_mask', labels_mask),
        ('add_mask', add_mask),
        ('add_index', add_index),
        ('dec_inputs', dec_inputs),
        ('dec_targets', dec_targets),
        ('nums_add', nums_add)
    ])
    self._token_start_indices = token_start_indices
    self.editing_task = task
    self._default_label = default_label

  def pad_to_max_length(self, max_seq_length, max_tgt_length, pad_token_id, add_index):
    """Pad the feature vectors so that they all have max_seq_length.

    Args:
      max_seq_length: The length that features will have after padding.
      pad_token_id: input_ids feature is padded with this ID, other features
        with ID 0.
    """
    pad_len = max_seq_length - len(self.features['input_ids'])
    for key in self.features:
      if key in ['add_index', 'nums_add']:
        continue
      pad_id = pad_token_id if key == 'input_ids' else 0
      if key == 'dec_inputs' or key == 'dec_targets':
        # for index in add_index:
        #   pad_tgt = max_tgt_length - len(self.features[key][index])
        #   self.features[key][index].extend([pad_id] * pad_tgt)
        for _ in range(pad_len):
          self.features[key].append([pad_id])
        for i in range(len(self.features[key])):
          pad_tgt = max_tgt_length - len(self.features[key][i])
          self.features[key][i].extend([pad_id] * pad_tgt)
      else:
        self.features[key].extend([pad_id] * pad_len)
        if len(self.features[key]) != max_seq_length:
          raise ValueError('{} has length {} (should be {}).'.format(
              key, len(self.features[key]), max_seq_length))

  def to_tf_example(self):
    """Returns this object as a tf.Example."""

    def int_feature(values):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    tf_features = collections.OrderedDict([
        (key, int_feature(val)) for key, val in self.features.items()
    ])
    return tf.train.Example(features=tf.train.Features(feature=tf_features))

  def get_token_labels(self):
    """Returns labels/tags for the original tokens, not for wordpieces."""
    labels = []
    for idx in self._token_start_indices:
      # For unmasked and untruncated tokens, use the label in the features, and
      # for the truncated tokens, use the default label.
      if (idx < len(self.features['labels']) and
          self.features['labels_mask'][idx]):
        labels.append(self.features['labels'][idx])
      else:
        labels.append(self._default_label)
    return labels


class BertExampleBuilder(object):
  """Builder class for BertExample objects."""

  def __init__(self, label_map, vocab_file,
               max_seq_length, max_tgt_length, do_lower_case,
               converter):
    """Initializes an instance of BertExampleBuilder.

    Args:
      label_map: Mapping from tags to tag IDs.
      vocab_file: Path to BERT vocabulary file.
      max_seq_length: Maximum sequence length.
      do_lower_case: Whether to lower case the input text. Should be True for
        uncased models and False for cased models.
      converter: Converter from text targets to tags.
    """
    self._label_map = label_map
    self._tokenizer = tokenization.FullTokenizer(vocab_file,
                                                 do_lower_case=do_lower_case)
    # tokenization.add_special_tokens({'additional_special_tokens': ["<e>"]})
    self._max_seq_length = max_seq_length
    self._max_tgt_length = max_tgt_length
    self._converter = converter
    self._pad_id = self._get_pad_id()
    self._keep_tag_id = self._label_map['KEEP']
    self._tags_only = {"KEEP": 1, "ADD": 4, "DELETE": 2, "SWAP": 3}

  def build_bert_example(
      self,
      sources,
      target = None,
      use_arbitrary_target_ids_for_infeasible_examples = False
  ):
    """Constructs a BERT Example.

    Args:
      sources: List of source texts.
      target: Target text or None when building an example during inference.
      use_arbitrary_target_ids_for_infeasible_examples: Whether to build an
        example with arbitrary target ids even if the target can't be obtained
        via tagging.

    Returns:
      BertExample, or None if the conversion from text to tags was infeasible
      and use_arbitrary_target_ids_for_infeasible_examples == False.
    """
    # Compute target labels.
    task = tagging.EditingTask(sources)
    flag = 1
    if target is not None:
      tags = self._converter.compute_tags(task, target)
      if not tags:
        flag = 0
        if use_arbitrary_target_ids_for_infeasible_examples:
          # Create a tag sequence [KEEP, DELETE, KEEP, DELETE, ...] which is
          # unlikely to be predicted by chance.
          tags = [tagging.Tag('KEEP') if i % 2 == 0 else tagging.Tag('DELETE')
                  for i, _ in enumerate(task.source_tokens)]
        else:
          return None, flag
    else:
      # If target is not provided, we set all target labels to KEEP.
      tags = [tagging.Tag('KEEP') for _ in task.source_tokens]
    # trans to our
    tags_only, add_mask, add_index, phrase, nums_add = self._converter.transtoadd(tags)
    labels_only = [self._tags_only[tag] for tag in tags_only]
    # labels = [self._label_map[str(tag)] for tag in tags]

    tokens, labels, token_start_indices, bert_add_mask, bert_phrase = self._split_to_wordpieces_only(
      task.source_tokens, labels_only, add_mask, phrase)
    # tokens, labels, token_start_indices = self._split_to_wordpieces(
    #     task.source_tokens, labels)

    tokens = self._truncate_list(tokens)
    labels = self._truncate_list(labels)
    add_mask = self._truncate_list(bert_add_mask)

    # 对phrase加bos, 同时dec_inputs and dec_targets
    dec_inputs = copy.deepcopy(bert_phrase)
    dec_targets = copy.deepcopy(bert_phrase)
    for i in range(len(bert_phrase)):
      if bert_phrase[i][0] != '[PAD]':
        dec_inputs[i].insert(0, '[BOS]')
        dec_targets[i].append('[SEP]')
    input_tokens = ['[CLS]'] + tokens + ['[SEP]']
    tmp_list = ['[PAD]']
    dec_inputs.insert(0, tmp_list)
    dec_inputs.append(tmp_list)
    dec_targets.insert(0, tmp_list)
    dec_targets.append(tmp_list)
    labels_mask = [0] + [1] * len(labels) + [0]
    labels = [0] + labels + [0]
    add_mask = [0] + add_mask + [0]
    # obtain add_index from add_mask
    add_index = []
    for i in range(len(add_mask)):
      if add_mask[i] == 1:
        add_index.append(i)

    input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    # phrase_ids = self._tokenizer.convert_tokens_to_ids(phrase)
    dec_inputs = [self._tokenizer.convert_tokens_to_ids(phrase) for phrase in dec_inputs]
    dec_targets = [self._tokenizer.convert_tokens_to_ids(phrase) for phrase in dec_targets]

    example = BertExample(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        labels=labels,
        labels_mask=labels_mask,
        token_start_indices=token_start_indices,
        task=task,
        default_label=self._keep_tag_id,
        add_mask=add_mask,
        add_index=add_index,
        dec_inputs=dec_inputs,
        dec_targets=dec_targets,
        nums_add=nums_add)
    example.pad_to_max_length(self._max_seq_length, self._max_tgt_length, self._pad_id, add_index)
    return example, flag, task

  def _split_to_wordpieces(self, tokens, labels):
    """Splits tokens (and the labels accordingly) to WordPieces.

    Args:
      tokens: Tokens to be split.
      labels: Labels (one per token) to be split.

    Returns:
      3-tuple with the split tokens, split labels, and the indices of the
      WordPieces that start a token.
    """
    bert_tokens = []  # Original tokens split into wordpieces.
    bert_labels = []  # Label for each wordpiece.
    # Index of each wordpiece that starts a new token.
    token_start_indices = []
    for i, token in enumerate(tokens):
      # '+ 1' is because bert_tokens will be prepended by [CLS] token later.
      token_start_indices.append(len(bert_tokens) + 1)
      pieces = self._tokenizer.tokenize(token)
      bert_tokens.extend(pieces)
      bert_labels.extend([labels[i]] * len(pieces))
    return bert_tokens, bert_labels, token_start_indices

  def _split_to_wordpieces_only(self, tokens, labels_only, add_mask, phrase):
    """Splits tokens (and the labels accordingly) to WordPieces.

    Args:
      tokens: Tokens to be split.
      labels: Labels (one per token) to be split.

    Returns:
      3-tuple with the split tokens, split labels, and the indices of the
      WordPieces that start a token.
    """
    bert_tokens = []  # Original tokens split into wordpieces.
    bert_labels = []  # Label for each wordpiece.
    bert_add_mask = []  # Add mask Label for each wordpiece.
    bert_phrase = []  # Phrase Label for each wordpiece.
    # Index of each wordpiece that starts a new token.
    token_start_indices = []
    for i, token in enumerate(tokens):
      # '+ 1' is because bert_tokens will be prepended by [CLS] token later.
      token_start_indices.append(len(bert_tokens) + 1)
      pieces = self._tokenizer.tokenize(token)
      bert_tokens.extend(pieces)
      bert_labels.extend([labels_only[i]] * len(pieces))
      bert_add_mask.extend([add_mask[i]] * len(pieces))
      bert_phrase.extend([phrase[i]] * len(pieces))
    for i in range(len(bert_phrase)):
      if bert_phrase[i] != '[PAD]':
        bert_phrase[i] = self._tokenizer.tokenize(bert_phrase[i])
      else:
        bert_phrase[i] = [bert_phrase[i]]
    return bert_tokens, bert_labels, token_start_indices, bert_add_mask, bert_phrase

  def _truncate_list(self, x):
    """Returns truncated version of x according to the self._max_seq_length."""
    # Save two slots for the first [CLS] token and the last [SEP] token.
    return x[:self._max_seq_length - 2]

  def _get_pad_id(self):
    """Returns the ID of the [PAD] token (or 0 if it's not in the vocab)."""
    try:
      return self._tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
    except KeyError:
      return 0
