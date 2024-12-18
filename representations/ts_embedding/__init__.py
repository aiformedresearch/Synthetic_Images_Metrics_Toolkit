# SPDX-License-Identifier: MIT
# Copyright (c) 2021, Ahmed M. Alaa, Boris van Breugel, Evgeny Saveliev, Mihaela van der Schaar

"""Timeseries encoding to a fixed size vector representation.

Author: Evgeny Saveliev (e.s.saveliev@gmail.com)
"""

from .seq2seq_autoencoder import Encoder, Decoder, Seq2Seq, init_hidden, compute_loss
from .training import train_seq2seq_autoencoder, iterate_eval_set
