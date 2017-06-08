# Copyright (c) 2017, Kyle Lo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def get_mu(eta1, eta2):
    return -0.5 * eta1 / eta2


def get_sigma(eta2):
    return np.sqrt(-0.5 / eta2)


def get_eta1(mu, sigma):
    return mu / sigma ** 2


def get_eta2(sigma):
    return -0.5 / sigma ** 2