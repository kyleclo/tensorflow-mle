# Copyright (c) 2017, Kyle Lo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def sfill(x, max_chars=10, justify='>'):
    """Fill a string with empty characters"""
    return '{}' \
        .format('{:' + justify + str(max_chars) + '}') \
        .format(x)


def sfloat(x, num_chars=10):
    """Stringify a float to have exactly some number of characters"""
    x = float(x)
    num_chars = int(num_chars)
    start, end = str(x).split('.')
    start_chars = len(str(float(start)))
    if start_chars > num_chars:
        raise Exception('Try num_chars = {}'.format(start_chars))
    return '{}' \
        .format('{:' + str(num_chars) + '.' +
                str(num_chars - start_chars + 1) + 'f}') \
        .format(x)


def shess(hess, num_chars=10):
    """Stringify an n x n Hessian matrix"""
    n = hess.shape[0]
    s = 'Hessian:' + ('\n' + '| {} ' * n + '|') * n
    return s.format(*[sfloat(h, num_chars)
                      for h in np.array(hess).reshape(-1)])


def sarray(x, num_chars=10):
    n = len(x)
    return '({})'.format(', '.join([sfloat(xi, num_chars / n) for xi in x]))
