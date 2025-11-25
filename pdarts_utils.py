
import numpy as np
import logging
import copy
from genotypes import PRIMITIVES
from genotypes import Genotype

def parse_network(switches_normal, switches_reduce):
    def _parse_switches(switches):
        n = 2
        start = 0
        gene = []
        step = 4
        for i in range(step):
            end = start + n
            for j in range(start, end):
                for k in range(len(switches[j])):
                    if switches[j][k]:
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene

    gene_normal = _parse_switches(switches_normal)
    gene_reduce = _parse_switches(switches_reduce)

    concat = range(2, 6)

    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )

    return genotype


def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1

    return index
def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index

def logging_switches(switches):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)

def check_sk_number(switches):
    count = 0
    for i in range(len(switches)):
        if switches[i][3]:
            count = count + 1

    return count

def delete_min_sk_prob(switches_in, switches_bk, probs_in):
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][3]:
            idx = -1
        else:
            idx = 0
            for i in range(3):
                if switches_bk[k][i]:
                    idx = idx + 1
        return idx
    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))]
    for i in range(len(switches_in)):
        idx = _get_sk_idx(switches_in, switches_bk, i)
        if not idx == -1:
            sk_prob[i] = probs_out[i][idx]
    d_idx = np.argmin(sk_prob)
    idx = _get_sk_idx(switches_in, switches_bk, d_idx)
    probs_out[d_idx][idx] = 0.0

    return probs_out


def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1

    return index


def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index


def logging_switches(switches):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)


def check_sk_number(switches):
    count = 0
    for i in range(len(switches)):
        if switches[i][3]:
            count = count + 1

    return count


def delete_min_sk_prob(switches_in, switches_bk, probs_in):
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][3]:
            idx = -1
        else:
            idx = 0
            for i in range(3):
                if switches_bk[k][i]:
                    idx = idx + 1
        return idx

    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))]
    for i in range(len(switches_in)):
        idx = _get_sk_idx(switches_in, switches_bk, i)
        if not idx == -1:
            sk_prob[i] = probs_out[i][idx]
    d_idx = np.argmin(sk_prob)
    idx = _get_sk_idx(switches_in, switches_bk, d_idx)
    probs_out[d_idx][idx] = 0.0

    return probs_out


def keep_1_on(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    for i in range(len(switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                idxs.append(j)
        drop = get_min_k_no_zero(probs[i, :], idxs, 2)
        for idx in drop:
            switches[i][idxs[idx]] = False
    return switches


def keep_2_branches(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    final_prob = [0.0 for i in range(len(switches))]
    for i in range(len(switches)):
        final_prob[i] = max(probs[i])
    keep = [0, 1]
    n = 3
    start = 2
    for i in range(3):
        end = start + n
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
        n = n + 1
    for i in range(len(switches)):
        if not i in keep:
            for j in range(len(PRIMITIVES)):
                switches[i][j] = False
    return switches