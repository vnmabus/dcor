from __future__ import absolute_import, division, print_function

import scipy.spatial


def _pdist(x, exponent=1):
    '''
    Pairwise distance between points in a set.
    '''

    metric = 'euclidean'

    if exponent != 1:
        metric = 'sqeuclidean'

    distances = scipy.spatial.distance.pdist(x, metric=metric)
    distances = scipy.spatial.distance.squareform(distances)

    if exponent != 1:
        distances **= exponent / 2

    return distances


def _cdist(x, y, exponent=1):
    '''
    Pairwise distance between the points in two sets.
    '''

    metric = 'euclidean'

    if exponent != 1:
        metric = 'sqeuclidean'

    distances = scipy.spatial.distance.cdist(x, y, metric=metric)

    if exponent != 1:
        distances **= exponent / 2

    return distances
