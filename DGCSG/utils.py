import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import random
from munkres import Munkres
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
from opt import args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def encode_labels(y):
    """Encode string labels to integers if they are not already numeric."""
    if isinstance(y[0], str):
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
    return y


def cluster_acc(y_true, y_pred):
    y_true = encode_labels(y_true)
    y_pred = encode_labels(y_pred)

    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)
    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]

        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    return acc

def eva(y_true, y_pred, epoch=0):

    acc=cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)

    return acc, nmi, ari

def normalization(features_):
    features = features_.copy()
    for i in range(len(features)):
        features[i] = features[i] / sum(features[i]) * 100000
    features = np.log2(features + 1)
    return features

def normalization_for_NE(features_):
    features = features_.copy()
    for i in range(len(features)):
        features[i] = features[i] / sum(features[i]) * 1000000
    features = np.log2(features + 1)
    return features

def getNeMatrix(W_in):
    N = len(W_in)

    K = min(20, N // 10)
    alpha = 0.9
    order = 3
    eps = 1e-20

    W0 = W_in * (1 - np.eye(N))
    W = NE_dn(W0, N, eps)
    W = (W + W.T) / 2

    DD = np.sum(np.abs(W0), axis=0)

    P = (dominateset(np.abs(W), min(K, N - 1))) * np.sign(W)
    P = P + np.eye(N) + np.diag(np.sum(np.abs(P.T), axis=0))

    P = TransitionFields(P, N, eps)

    D, U = np.linalg.eig(P)
    d = D - eps
    d = (1 - alpha) * d / (1 - alpha * d ** order)
    D = np.diag(d)
    W = np.dot(np.dot(U, D), U.T)
    W = (W * (1 - np.eye(N))) / (1 - np.diag(W))
    W = W.T

    D = np.diag(DD)
    W = np.dot(D, W)
    W[W < 0] = 0
    W = (W + W.T) / 2

    return W

#确定每个样本的k近邻
def dominateset(aff_matrix, NR_OF_KNN):
    thres = np.sort(aff_matrix)[:, -NR_OF_KNN]
    aff_matrix.T[aff_matrix.T < thres] = 0
    aff_matrix = (aff_matrix + aff_matrix.T) / 2
    return aff_matrix
#计算转移矩阵
def TransitionFields(W, N, eps):
    W = W * N
    W = NE_dn(W, N, eps)
    w = np.sqrt(np.sum(np.abs(W), axis=0) + eps)
    W = W / np.expand_dims(w, 0).repeat(N, 0)
    W = np.dot(W, W.T)
    return W
def NE_dn(w, N, eps):
    w = w * N
    D = np.sum(np.abs(w), axis=1) + eps
    D = 1 / D
    D = np.diag(D)
    wn = np.dot(D, w)
    return wn
"""
Construct a graph based on the cell features
"""


def getGraph(dataset_str, features, L, K, method):

    if method == 'pearson':
        co_matrix = np.corrcoef(features)
    elif method == 'spearman':
        co_matrix, _ = spearmanr(features.T)
    elif method == 'NE':
        co_matrix = np.corrcoef(features)

        NE_path = 'result/NE_' + dataset_str + '.csv'
        if os.path.exists(NE_path):
            NE_matrix = pd.read_csv(NE_path).values
        else:
            features = normalization_for_NE(features)
            in_matrix = np.corrcoef(features)
            NE_matrix = getNeMatrix(in_matrix)
           # pd.DataFrame(NE_matrix).to_csv(NE_path, index=False)
        N = len(co_matrix)
        sim_sh = 1.
        for i in range(len(NE_matrix)):
            NE_matrix[i][i] = sim_sh * max(NE_matrix[i])

        data = NE_matrix.reshape(-1)
        data = np.sort(data)
        data = data[:-int(len(data) * 0.02)]

        min_sh = data[0]
        max_sh = data[-1]

        delta = (max_sh - min_sh) / 100

        temp_cnt = []
        for i in range(20):
            s_sh = min_sh + delta * i
            e_sh = s_sh + delta
            temp_data = data[data > s_sh]
            temp_data = temp_data[temp_data < e_sh]
            temp_cnt.append([(s_sh + e_sh) / 2, len(temp_data)])

        candi_sh = -1
        for i in range(len(temp_cnt)):
            pear_sh, pear_cnt = temp_cnt[i]
            if 0 < i < len(temp_cnt) - 1:
                if pear_cnt < temp_cnt[i + 1][1] and pear_cnt < temp_cnt[i - 1][1]:
                    candi_sh = pear_sh
                    break
        if candi_sh < 0:
            for i in range(1, len(temp_cnt)):
                pear_sh, pear_cnt = temp_cnt[i]
                if pear_cnt * 2 < temp_cnt[i - 1][1]:
                    candi_sh = pear_sh
        if candi_sh == -1:
            candi_sh = 0.3

        propor = len(NE_matrix[NE_matrix <= candi_sh]) / (len(NE_matrix) ** 2)
        propor = 1 - propor
        thres = np.sort(NE_matrix)[:, -int(len(NE_matrix) * propor)]
        co_matrix.T[NE_matrix.T <= thres] = 0

    else:
        return

    N = len(co_matrix)

    up_K = np.sort(co_matrix)[:, -K]

    mat_K = np.zeros(co_matrix.shape)
    mat_K.T[co_matrix.T >= up_K] = 1

    thres_L = np.sort(co_matrix.flatten())[-int(((N * N) // (1 // (L + 1e-8))))]
    mat_K.T[co_matrix.T < thres_L] = 0

    return mat_K

def calculate_dropout_rate_np(x):
    """
    计算 NumPy 数组 x 的丢包率（即 0 元素占总元素的比例）
    参数:
        x: 输入的 NumPy 数组
    返回:
        dropout_rate: 丢包率
    """
    zero_elements = np.sum(x == 0.0)  # 统计 0 元素的数量
    total_elements = x.size  # 总元素数量
    dropout_rate = zero_elements / total_elements  # 计算丢包率
    print(f"Dropout Rate: {dropout_rate:.2%}")
    return dropout_rate
"""
Load scRNA-seq data set and perfrom preprocessing
"""

def load_data(data_path, dataset_str, PCA_dim, is_NE=True, n_clusters=20):
    # Get data
    DATA_PATH = data_path

    data = pd.read_csv(DATA_PATH, index_col=0, sep='\t')
    cells = data.columns.values
    genes = data.index.values
    features = data.values.T

    # Preprocess features
    features = normalization(features)

    # Construct graph
    N = len(cells)
    avg_N = N // n_clusters
    K = avg_N // 10
    K = min(K, 20)
    K = max(K, 6)

    L = 0
    if is_NE:
        method = 'NE'
    else:
        method = 'pearson'
    adj = getGraph(dataset_str, features, L, K, method)

    # feature tranformation
    if features.shape[0] > PCA_dim and features.shape[1] > PCA_dim:
        pca = PCA(n_components=PCA_dim)
        features = pca.fit_transform(features)
    else:
        var = np.var(features, axis=0)
        min_var = np.sort(var)[-1 * PCA_dim]
        features = features.T[var >= min_var].T
        features = features[:, :PCA_dim]
    print('Shape after transformation:', features.shape)

    features = (features - np.mean(features)) / (np.std(features))

    true_path = '.\data\{}\\new_label.ann'.format(args.name)
    if dataset_str == "Klein":
        true = pd.read_csv(true_path, sep=',').values
    else:
        true = pd.read_csv(true_path, sep='\t').values
    cells = true[:, 0]

    y = true[:, -1].astype(str)
    return adj, features,y, cells, genes
