# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A checkpointing mechanism for Dopamine agents.

This Checkpointer expects a base directory where checkpoints for different
iterations are stored. Specifically, Checkpointer.save_checkpoint() takes in
as input a dictionary 'data' to be pickled to disk. At each iteration, we
write a file called 'cpkt.#', where # is the iteration number. The
Checkpointer also cleans up old files, maintaining up to the
`checkpoint_duration` most recent iterations.

The Checkpointer writes a sentinel file to indicate that checkpointing was
globally successful. This means that all other checkpointing activities
(saving the Tensorflow graph, the replay buffer) should be performed *prior*
to calling Checkpointer.save_checkpoint(). This allows the Checkpointer to
detect incomplete checkpoints.

#### Example

After running 10 iterations (numbered 0...9) with base_directory='/checkpoint',
the following files will exist:
```
  /checkpoint/cpkt.6
  /checkpoint/cpkt.7
  /checkpoint/cpkt.8
  /checkpoint/cpkt.9
  /checkpoint/sentinel_checkpoint_complete.6
  /checkpoint/sentinel_checkpoint_complete.7
  /checkpoint/sentinel_checkpoint_complete.8
  /checkpoint/sentinel_checkpoint_complete.9
```
Checkpoint management for Dopamine agents.

The Checkpointer uses a base directory containing data for different iterations.
Checkpointer.save_checkpoint() serializes its data argument to a file named 'ckpt.#',
where # is the iteration number. Older files are removed to limit retention.

A sentinel file marks a checkpoint as complete. Other checkpoint operations, including
saving the TensorFlow graph and replay buffer, must finish before save_checkpoint().

#### Example
After ten iterations (0...9), the base directory '/checkpoint' contains:
```
  /checkpoint/cpkt.6
  /checkpoint/cpkt.7
  /checkpoint/cpkt.8
  /checkpoint/cpkt.9
  /checkpoint/sentinel_checkpoint_complete.6
  /checkpoint/sentinel_checkpoint_complete.7
  /checkpoint/sentinel_checkpoint_complete.8
  /checkpoint/sentinel_checkpoint_complete.9
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

from absl import logging

import gin
import tensorflow as tf


@gin.configurable
def get_latest_checkpoint_number(base_directory,
                                 override_number=None,
                                 sentinel_file_identifier='checkpoint'):
  """Returns the version number of the latest completed checkpoint.

  Args:
    base_directory: str, directory in which to look for checkpoint files.
    override_number: None or int, allows the user to manually override
      the checkpoint number via a gin-binding.
    sentinel_file_identifier: str, prefix used by checkpointer for naming
      sentinel files.

  Returns:
    int, the iteration number of the latest checkpoint, or -1 if none was found.
  """
  if override_number is not None:
    return override_number

  sentinel = 'sentinel_{}_complete.*'.format(sentinel_file_identifier)
  glob = os.path.join(base_directory, sentinel)
  def extract_iteration(x):
    return int(x[x.rfind('.') + 1:])
  try:
    checkpoint_files = tf.io.gfile.glob(glob)
  except tf.errors.NotFoundError:
    return -1
  try:
    latest_iteration = max(extract_iteration(x) for x in checkpoint_files)
    return latest_iteration
  except ValueError:
    return -1


@gin.configurable
class Checkpointer(object):
  """
  Class for managing checkpoints for Dopamine agents.
  Manages checkpoints for Dopamine agents.
  """

  def __init__(self, base_directory, checkpoint_file_prefix='ckpt',
               sentinel_file_identifier='checkpoint', checkpoint_frequency=1,
               checkpoint_duration=4,
               keep_every=None):
    """Initializes Checkpointer.

    Args:
      base_directory: str, directory where all checkpoints are saved/loaded.
      checkpoint_file_prefix: str, prefix to use for naming checkpoint files.
      sentinel_file_identifier: str, prefix to use for naming sentinel files.
      checkpoint_frequency: int, the frequency at which to checkpoint.
      checkpoint_duration: int, how many checkpoints to keep
      keep_every: Optional (int or None), keep all checkpoints == 0 % this
        number. Set to None to disable.

    Raises:
      ValueError: if base_directory is empty, or not creatable.

    Initializes a Checkpointer.

    Args:
      base_directory: str,
        Directory used to save and load checkpoint files.
      checkpoint_file_prefix: str,
        Prefix for checkpoint filenames.
      sentinel_file_identifier: str,
        Prefix for sentinel filenames.
      checkpoint_frequency: int,
        Frequency of checkpoint saves.
      checkpoint_duration: int,
        Number of recent checkpoints to retain.
      keep_every: int, optional=None,
        Retain checkpoints at this interval indefinitely; None disables this option.
    """
    # A base directory is required.
    if not base_directory:
      raise ValueError('No path provided to Checkpointer.')
    # Checkpoint filename prefix.
    self._checkpoint_file_prefix = checkpoint_file_prefix
    # Sentinel filename prefix.
    self._sentinel_file_prefix = 'sentinel_{}_complete'.format(
        sentinel_file_identifier)
    # Checkpoint save frequency.
    self._checkpoint_frequency = checkpoint_frequency
    # Number of recent checkpoints to retain.
    self._checkpoint_duration = checkpoint_duration
    # Interval for checkpoints retained indefinitely.
    self._keep_every = keep_every
    # Base checkpoint directory.
    self._base_directory = base_directory
    # Create the base directory.
    try:
      tf.io.gfile.makedirs(base_directory)
    except tf.errors.PermissionDeniedError as permission_error:
      # We catch the PermissionDeniedError and issue a more useful exception.
      raise ValueError('Unable to create checkpoint path: {}.'.format(
          base_directory)) from permission_error

  def _generate_filename(self, file_prefix, iteration_number) -> str:
    """
    Returns a checkpoint filename from prefix and iteration number.
    Returns a checkpoint filename from a prefix and iteration number.
    Args:
      file_prefix: str,
        Checkpoint filename prefix.
      iteration_number: int,
        Iteration number.
    Return: str,
      Checkpoint filename.
    """
    filename = '{}.{}'.format(file_prefix, iteration_number)
    return os.path.join(self._base_directory, filename)

  def _save_data_to_file(self, data, filename):
    """
    Saves the given 'data' object to a file.
    Saves the supplied data to a file.
    """
    with tf.io.gfile.GFile(filename, 'w') as fout:
      pickle.dump(data, fout)

  def save_checkpoint(self, iteration_number, data):
    """Saves a new checkpoint at the current iteration_number.

    Args:
      iteration_number: int, the current iteration number for this checkpoint.
      data: Any (picklable) python object containing the data to store in the
        checkpoint.
    """
    # Save only at the configured iteration frequency.
    if iteration_number % self._checkpoint_frequency != 0:
      return
    # Construct the checkpoint filename.
    filename = self._generate_filename(self._checkpoint_file_prefix,
                                       iteration_number)
    # Write checkpoint data to the destination file.
    self._save_data_to_file(data, filename)
    # Construct the sentinel filename.
    filename = self._generate_filename(self._sentinel_file_prefix,
                                       iteration_number)
    # Mark the checkpoint as successfully completed.
    with tf.io.gfile.GFile(filename, 'wb') as fout:
      fout.write('done')
    # Remove older checkpoints.
    self._clean_up_old_checkpoints(iteration_number)

  def _clean_up_old_checkpoints(self, iteration_number):
    """Removes sufficiently old checkpoints."""
    # After writing a the checkpoint and sentinel file, we garbage collect files
    # that are self._checkpoint_duration * self._checkpoint_frequency
    # versions old.
    stale_iteration_number = iteration_number - (self._checkpoint_frequency *
                                                 self._checkpoint_duration)

    # If keep_every has been set, we spare every keep_every'th checkpoint
    if (self._keep_every is not None
        and (stale_iteration_number %
             (self._keep_every*self._checkpoint_frequency) == 0)):
      return

    if stale_iteration_number >= 0:
      stale_file = self._generate_filename(self._checkpoint_file_prefix,
                                           stale_iteration_number)
      stale_sentinel = self._generate_filename(self._sentinel_file_prefix,
                                               stale_iteration_number)
      try:
        tf.io.gfile.remove(stale_file)
        tf.io.gfile.remove(stale_sentinel)
      except tf.errors.NotFoundError:
        # Ignore if file not found.
        logging.info('Unable to remove %s or %s.', stale_file, stale_sentinel)

  def _load_data_from_file(self, filename):
    if not tf.io.gfile.exists(filename):
      return None
    with tf.io.gfile.GFile(filename, 'rb') as fin:
      return pickle.load(fin)

  def load_checkpoint(self, iteration_number):
    """Tries to reload a checkpoint at the selected iteration number.

    Args:
      iteration_number: The checkpoint iteration number to try to load.

    Returns:
      If the checkpoint files exist, two unpickled objects that were passed in
        as data to save_checkpoint; returns None if the files do not exist.
    """
    checkpoint_file = self._generate_filename(self._checkpoint_file_prefix,
                                              iteration_number)
    return self._load_data_from_file(checkpoint_file)
