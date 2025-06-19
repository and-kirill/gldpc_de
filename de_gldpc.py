"""
This module implements a quantized density evolution over a proto-graph given a hard BPSK input
"""
import os
import ctypes
import time

import numpy as np
from scipy.special import erf
import galois

GROUP = 3

class QuantizedPDF:
    """
    Quantized probability density function distribution
    PDF is represented by a vector of length 2N + 1, with elements corresponding to:
     - the first N elements represent negative LLRs in increasing order (decreasing magnitude)
     - N+1-st point corresponds to zero LLR
     - Remaining points correspond to positive LLR values in increasing order
    LLR value = LLR index - N
    LLR index = LLR value + N
    """
    def __init__(self, lib, n_points, pdf=None):
        """
        Initialized Quantized PDF with the following parameters:
        :param lib: Shared library with quantized DE functions implementation
        :param n_points: the number of points N
        :param pdf: vector of probabilities (aux, used by copy() only)
        """
        self.n_points = n_points
        if pdf is None:
            self.pdf = np.zeros(2 * n_points + 1)
        else:
            self.pdf = np.copy(pdf)
        self.lib = lib

    def min_sum_abs(self, other):
        """
        Apply a pairwise min-sum rule (check node operation)
        :param other: PDF to be considered in min-sum rule
        :return: None, just updates internal state as self = MIN_SUM(self, other)
        """
        pdf_out = np.zeros_like(self.pdf)
        self.lib.minsum_abs(self.pdf, other.pdf, pdf_out, self.n_points)
        self.pdf = pdf_out

    def convolve(self, other):
        """
        Apply a pairwise convolution rule (variable node operation)
        :param other: PDF to be considered in convolution
        :return: None, just updates internal state as self = CONVOLVE(self, other)
        """
        pdf_out = np.zeros_like(self.pdf)
        self.lib.convolve_thr(self.pdf, other.pdf, pdf_out, self.n_points)
        self.pdf = pdf_out

    def node_op(self, other, node_type):
        """
        Switcher between convolution/min-sum to unify Tanner graph processing functionality
        :param other PDF to be considered in node operation
        :param node_type: string, supported values are 'check' or 'variable
        """
        if node_type == 'check':
            self.min_sum_abs(other)
        elif node_type == 'variable':
            self.convolve(other)
        else:
            raise ValueError('Unknown node operation')

    def normalize(self):
        """
        Force probabilities to have unit sum
        :return: None, updates internal state
        """
        self.pdf = self.pdf / np.sum(self.pdf)

    def scale(self, llr_scale):
        """
        Apply a scale (for the normalized min-sum)
        :param llr_scale: LLR scale, double value
        :return: None, just updates internal state
        """
        pdf_out = np.zeros_like(self.pdf)
        self.lib.scale(self.pdf, pdf_out, self.n_points, llr_scale)
        self.pdf = pdf_out
        self.normalize()

    def gldpc_cn_op(self, other1):
        """
        Apply a pairwise convolution rule (variable node operation)
        :param other: PDF to be considered in convolution
        :return: None, just updates internal state as self = CONVOLVE(self, other)
        """
        pdf_out = np.zeros_like(self.pdf)
        self.lib.gldpc_cn_op(self.pdf, other1.pdf, pdf_out, self.n_points)
        self.pdf = pdf_out

    def copy(self):
        """
        Make a copy
        :return: copied object
        """
        return QuantizedPDF(self.lib, self.n_points, self.pdf)

    def delta(self):
        """
        Return PDF concentrated at zero (delta function)
        """
        pdf_out = np.zeros_like(self.pdf)
        pdf_out[len(pdf_out) // 2] = 1
        return QuantizedPDF(self.lib, self.n_points, pdf_out)


SRC_FILE = 'de_gldpc_impl'


def lib_compile():
    """
    Compile C++ code
    """
    wdir = os.path.dirname(__file__)
    src_abs = os.path.join(wdir, SRC_FILE)
    os.system(f'g++ -g3 -Wall -Wextra -O3 -fPIC -c -o {src_abs}.o      {src_abs}.cpp')
    os.system(f'g++ -g3  -Wall -Wextra -shared     -o {src_abs}_lib.so {src_abs}.o  ')
    os.system('rm ' + wdir + '/*.o')


def load_lib():
    """
    Load C++ implementation
    :return: ctypes CDLL lib
    """
    wdir = os.path.dirname(__file__)
    src_abs = os.path.join(wdir, SRC_FILE)
    lib = ctypes.CDLL(src_abs + '_lib.so')
    # Min-sum rule (check node operations)
    lib.minsum_abs.restype = None
    lib.minsum_abs.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64),  # PMF1
        np.ctypeslib.ndpointer(dtype=np.float64),  # PMF2
        np.ctypeslib.ndpointer(dtype=np.float64),  # PMF out
        ctypes.c_uint,  # The number of points
    ]
    # Convolution rule with saturation (variable node operations)
    lib.convolve_thr.restype = None
    lib.convolve_thr.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64),  # PMF1
        np.ctypeslib.ndpointer(dtype=np.float64),  # PMF2
        np.ctypeslib.ndpointer(dtype=np.float64),  # PMF out
        ctypes.c_uint,  # The number of points
    ]
    # Scaling function
    lib.scale.restype = None
    lib.scale.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64),  # PMF in
        np.ctypeslib.ndpointer(dtype=np.float64),  # PMF out
        ctypes.c_uint,  # The number of points
        ctypes.c_double,  # LLR scale
    ]
    lib.gldpc_cn_op.restype = None
    lib.gldpc_cn_op.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64),  # PMF1
        np.ctypeslib.ndpointer(dtype=np.float64),  # PMF2
        np.ctypeslib.ndpointer(dtype=np.float64),  # PMF out
        ctypes.c_uint,  # The number of points
    ]

    return lib


class MessageIndexer:
    """
    Message indexing for a parity check matrix.
    Get a list of messages adjacent to particular variable/check nodes
    """
    def __init__(self, pcm):
        """
        Initialize message indices using per-row sequential indexing
        :param pcm: parity check matrix (2D numpy array of type np.uint8)
        """
        self.pcm = pcm
        self.pcm_idx = self.pcm.astype(np.int32)
        pcm_nz = int(np.sum(pcm))
        self.pcm_idx[self.pcm_idx > 0] = np.arange(1, pcm_nz + 1)

    def adjacent_to_vn(self, vn_index):
        """
        Get a list of vertices adjacent to variable node
        :param vn_index: variable node index (from 0 to block length - 1)
        :return: a list of _messages_
        """
        column = self.pcm_idx[:, vn_index] - 1
        return column[column >= 0]

    def adjacent_to_cn(self, cn_index):
        """
        Get a list of vertices adjacent to variable node
        :param cn_index: check node index (from 0 to number of checks - 1)
        :return: a list of _messages_
        """
        row = self.pcm_idx[cn_index, :] - 1
        return row[row >= 0]

def norm_cdf(x_series, mu, sigma):
    cdf_values = erf((x_series - mu) / (np.sqrt(2)*sigma)) / 2
    return cdf_values

#generate a normal distribution for a awgn channel
def init_awgn_channel(lib, settings):
    scale = settings.input_scale
    max_llr = settings.max_llr
    n_points = scale * max_llr
    sigma = np.sqrt(0.5 * 10**(-settings.snr / 10))
    x_series = np.arange(-max_llr, max_llr, 1 / scale)
    pdf = QuantizedPDF(lib, n_points)
    m_channel = 2 / sigma**2
    cdf_values = norm_cdf(x_series - 1 / (2 * scale), m_channel, np.sqrt(2 * m_channel)) + 0.5
    pdf.pdf = (np.append(cdf_values, 1) - np.insert(cdf_values, 0, 0))
    return pdf

#distribution for bsc channel
def init_channel(lib, settings):
    """
    Initialize chanel LLR:
    non-zero PDF at +/- scale with [1 - pe_channel, pe_channel] values
    :param lib: QuantizedDE ctypes impl
    :return: QuantizedPDF instance
    """
    pe_channel = settings.exp.pe_channel  # BSC cross-over probability
    scale = settings.de.input_scale  # Input LLR scale: +/-1 will be converted to +/-scale
    # Saturation value. PDF will have 2N + 1 points, N = scale * max_llr
    max_llr = settings.de.max_llr

    n_points = scale * max_llr
    pdf = QuantizedPDF(lib, n_points)
    pdf.pdf[n_points - scale] = pe_channel
    pdf.pdf[n_points + scale] = 1.0 - pe_channel
    return pdf


def init_erasure(lib, settings):
    """
    Initialize erased LLR: all density is concentrated at zero LLR
    :param lib: QuantizedDE ctypes impl
    :return: QuantizedPDF instance
    """
    n_points = settings.de.max_llr * settings.de.input_scale
    pdf = QuantizedPDF(lib, n_points)
    pdf.pdf[n_points] = 1.0
    return pdf


class TannerNode:
    """
    Tanner graph common node operation implementation.
    The main purpose is to generate output distribution being passed over
    particular edge of the Tanner graph.

    The output message is a result of node operations taken over all input messages
    except the outgoing edge.

    To reduce the numerical complexity, run a sequence of node operations
    from left-to-right and from right to left. Next, when deriving a particular
    distribution, take two distributions from left-to-right and right-to-left sequences.
    """
    def __init__(self, pmf_list, index_list, node_type):
        """
        Initialize check node using a list of input distributions and a list of message indices
        :param pmf_list: List of input distributions
        :param index_list: List of edge indices corresponding to these distributions
        """
        self.node_type = node_type
        self.left2right = self.__left2right(pmf_list)
        self.right2left = self.__right2left(pmf_list)
        self.index_list = index_list
        self.degree = len(pmf_list)

    def get(self, message_id):
        """
        Get the output distribution for corresponding edge index
        :param message_id: Output edge index
        :return: resulting PMF
        """
        if self.degree == 0:
            raise ValueError('Degree-0 nodes are not processed')
        if self.degree == 1:
            return self.right2left[0].delta()

        index = self.index_list.index(message_id)
        if index == 0:
            result = self.right2left[0].copy()
        elif index >= len(self.index_list) - 1:
            result = self.left2right[-1].copy()
        else:
            result = self.left2right[index - 1].copy()
            result.node_op(self.right2left[index], self.node_type)
        return result

    def __left2right(self, pmf_list):
        """
        Left-to-right pass
        :param pmf_list: LLR distributions incoming to this Tanner graph node
        :return: a sequence of node operations (from the left to the right) applied to
                 input distributions
        """
        left2right = [pmf_list[0].copy()]
        for i in range(1, len(pmf_list) - 1):
            pdf_cur = left2right[-1].copy()
            pdf_cur.node_op(pmf_list[i], self.node_type)
            left2right.append(pdf_cur)
        return left2right

    def __right2left(self, pmf_list):
        """
        Right-to-left pass
        :param pmf_list: input distributions
        :return: a sequence of node operations (from the right to the left)
        """
        right2left = [pmf_list[-1].copy()]
        for i in reversed(range(1, len(pmf_list) - 1)):
            pdf_cur = right2left[0].copy()
            pdf_cur.node_op(pmf_list[i], self.node_type)
            right2left.insert(0, pdf_cur)
        return right2left
    
class GeneralizedCheckNode:
    """
    Tanner graph Check node operation implementation.
    The main purpose is to generate output distribution being passed over
    particular edge of the Tanner graph.

    The output message is a result of node operations taken over all input messages
    except the outgoing edge.

    To reduce the numerical complexity, run a sequence of node operations
    from left-to-right and from right to left. Next, when deriving a particular
    distribution, take two distributions from left-to-right and right-to-left sequences.
    """
    def __init__(self, pmf_list, index_list, node_type):
        """
        Initialize check node using a list of input distributions and a list of message indices
        :param pmf_list: List of input distributions
        :param index_list: List of edge indices corresponding to these distributions
        """
        self.node_type = node_type
        self.index_list = index_list
        self.degree = len(pmf_list)
        self.grouped_pmf_list = [[], [], []]
        self.grouped_index_list = [[], [], []]
        for i in range(len(pmf_list)):
            idx = i % GROUP
            self.grouped_pmf_list[idx].append(pmf_list[i])
            self.grouped_index_list[idx].append(index_list[i])
        self.tanner_nodes = [TannerNode(self.grouped_pmf_list[i], self.grouped_index_list[i], self.node_type) for i in range(GROUP)]
        self.left2rightend = self.__left2rightend()
        self.skip_group_sum = self.__skip_group_sum()

    def get(self, message_id):
        """
        Get the output distribution for corresponding edge index
        :param message_id: Output edge index
        :return: resulting PMF
        """
        index = self.index_list.index(message_id)
        group = index % GROUP

        res = self.tanner_nodes[group].get(message_id)
        res.gldpc_cn_op(self.skip_group_sum[group])
        return res
    
    def __left2rightend(self):
        """
        Get the end to end per group convolution  
        """
        out = []
        for i in range(GROUP):
            temp = self.tanner_nodes[i].left2right[-1].copy()
            temp.node_op(self.grouped_pmf_list[i][-1], self.node_type)
            out.append(temp.copy())
        return out
    
    def __skip_group_sum(self):
        """
        Function evaluates distribution of sum of two groups except the group in question
        """
        out = []
        for i in range(GROUP):
            groups = np.setdiff1d(np.arange(GROUP), i)
            temp = self.left2rightend[groups[0]].copy()
            temp.convolve(self.left2rightend[groups[1]])
            out.append(temp.copy())
        return out

class QuantizedDE:
    """
    Quantized DE implementation
    """
    def __init__(self, lib, pcm, settings):
        self.lib = lib
        self.indexer = MessageIndexer(pcm)
        self.settings = settings
        self.punc_idx = np.arange(settings.punctured).tolist()
        self.pcm = pcm
        self.n_checks, self.blocklen = pcm.shape

        self.channel_msg = {}
        for vn_index in range(self.blocklen):
            if vn_index in self.punc_idx:
                self.channel_msg[vn_index] = init_erasure(lib, settings)
            else:
                self.channel_msg[vn_index] = init_awgn_channel(lib, settings)
        # Messages from variable to check nodes. Must copy corresponding intput distributions
        # in accordance with the parity check matrix
        self.q_msg = {}
        # Messages from check to variable nodes. Initialized with a kind of empty distribution
        self.r_msg = {}

    def init_distributions(self):
        """
        copy channel messages to proper Q-messages
        """
        for vn_index in range(self.blocklen):
            for j in self.indexer.adjacent_to_vn(vn_index):
                self.q_msg[j] = self.channel_msg[vn_index].copy()
    
    def ber_per_vn(self):
        """
        Derive per-symbol error probabilities
        :return: list of per-variable node error probabilities
        """
        error_probs = np.zeros(self.blocklen,)
        for vn_index in range(self.blocklen):
            llr_distr = self.channel_msg[vn_index].copy()
            adjacent_nodes = self.indexer.adjacent_to_vn(vn_index)
            for j in adjacent_nodes:
                llr_distr.convolve(self.r_msg[j].copy())
            error_probs[vn_index] = np.sum(llr_distr.pdf[:(llr_distr.n_points + 1)])
        return error_probs

    def information_set(self, ber_per_vn):
        pcm_bin = binarize_pcm(self.pcm)
        # Parity set indices
        # Corresponding columns of the parity check matrix must for a full-rank matrix
        parity_idx = []
        m_parity = np.min(pcm_bin.shape)
        for col_id in np.argsort(-ber_per_vn):  # Sort estimated BER in decreasing order
            # Do not add punctured bits into parity set
            if col_id in self.punc_idx:
                continue
            if gf2_rank(pcm_bin[:, parity_idx + [col_id]]) > len(parity_idx):
                parity_idx.append(col_id)
            if len(parity_idx) == m_parity:
                break

        # Information set is a set of non-parity indices
        return np.setdiff1d(np.arange(self.blocklen), parity_idx)

    def output_ber(self):
        """
        Calculate optimization metric
        """
        ber_per_vn = self.ber_per_vn()
        inf_idx = self.information_set(ber_per_vn)
        # Return mean BER over the information set
        return np.mean(ber_per_vn[inf_idx])

    def plot_pdfs(self):
        out_pdfs = self.output_distr()
        for i in range(pcm.shape[1]):
            plt.plot(out_pdfs[i, :])
        plt.show()

    def output_distr(self):
        out_pdfs = []
        for vn_index in range(self.blocklen):
            llr_distr = self.channel_msg[vn_index].copy()
            adjacent_nodes = self.indexer.adjacent_to_vn(vn_index)
            for j in adjacent_nodes:
                llr_distr.convolve(self.r_msg[j].copy())
            out_pdfs.append(llr_distr.pdf)
        return np.vstack(out_pdfs)

    def de_min_sum(self):
        """
        Run min-sum-based density evolution
        :return: per-variable node error probabilities
        """
        self.init_distributions()
        for i in range(self.settings.n_iterations):
            self.de_min_sum_iteration(self.settings.llr_scale)
        return self.output_ber()

    def de_min_sum_iteration(self, llr_scale):
        """
        Run single iteration of density evolution
        :param llr_scale: the scale of LLRs (similar to decoding algorithm)
        :return: None
        """
        # Update R-messages
        for cn_index in range(self.n_checks):
            adjacent_nodes = self.indexer.adjacent_to_cn(cn_index).tolist()
            cnop = GeneralizedCheckNode(
                [self.q_msg[i] for i in adjacent_nodes],
                adjacent_nodes,
                node_type='check'
            )
            for j in adjacent_nodes:
                self.r_msg[j] = cnop.get(j)
                self.r_msg[j].scale(llr_scale)
                
        # Update Q-messages
        for vn_index in range(self.blocklen):
            adjacent_nodes = self.indexer.adjacent_to_vn(vn_index).tolist()
            vnop = TannerNode(
                # first convert from check to variable message quatization points to perform operation
                [self.r_msg[i] for i in adjacent_nodes],
                adjacent_nodes,
                node_type='variable'
            )
            for j in adjacent_nodes:
                self.q_msg[j] = vnop.get(j)
                self.q_msg[j].convolve(self.channel_msg[vn_index])


def binarize_pcm(pcm_gldpc):
    """
    Expand the GLDPC parity check matrix by Cordaro-Wagner code
    """
    # Cordaro-Wagner extension sequence
    seq_len = int(np.ceil(np.max(np.sum(pcm_gldpc, 1)) / 3))
    cw_seq = np.hstack([np.array([[1, 0, 1], [0, 1, 1]])] * seq_len)

    pcm_bin = np.zeros((pcm_gldpc.shape[0] * 2, pcm_gldpc.shape[1]))
    for i in range(pcm_gldpc.shape[0]):
        idx = np.argwhere(pcm_gldpc[i, :]).reshape(-1)
        pcm_bin[(2 * i):(2 * i + 2), idx] = cw_seq[:, :len(idx)]

    return pcm_bin.astype(np.uint8)


def gf2_rank(mtx):
    """
    Wrapper for matrix rank in GF2
    """
    return np.linalg.matrix_rank(galois.GF2(mtx))


if __name__ == '__main__':
    import json
    from settings import Settings
    import matplotlib.pyplot as plt
    from simulator_awgn_python.channel import AwgnQAMChannel

    lib_compile()

    with open('optimization.json', 'r', encoding='utf-8') as fhandle:
        config = json.load(fhandle)
    de_settings = Settings('optimization.json')
    pcm = np.loadtxt(config['exp']['init_point'], dtype=np.uint8)

    print(de_settings)
    lib = load_lib()
    qde = QuantizedDE(lib, pcm, de_settings)
    ber_per_vn = qde.de_min_sum()
    print('Max BER estimate:', np.max(ber_per_vn))
    out_pdfs = qde.output_distr()
    for i in range(pcm.shape[1]):
        plt.plot(out_pdfs[i, :])
    plt.show()