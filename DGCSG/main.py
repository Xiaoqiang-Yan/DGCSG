import torch
import os
import pandas as pd

import plotly.graph_objects as go
from torch.optim import Adam, AdamW
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F
from DGCSG import DGCSG
from utils import setup_seed, target_distribution, eva, LoadDataset, load_data
from opt import args
import datetime
from logger import Logger, metrics_info, record_info

nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')

def train(model, x, y):

    acc_reuslt = []
    nmi_result = []
    ari_result = []
    loss_list = []
    original_acc = -1
    metrics = [' nmi', ' ari']
    logger = Logger(args.name+ '==' + nowtime)
    logger.info(model)
    logger.info(args)
    logger.info(metrics_info(metrics))
    save_dir = './epoch_labels'
    os.makedirs(save_dir, exist_ok=True)

    n_clusters = args.n_clusters
    optimizer = AdamW(model.parameters(), lr=args.lr)
    with torch.no_grad():
        z, _, _, _ = model.ae(x)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    cluster_id = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    for epoch in range(400):
        x_bar,  adj_hat, z_ae, q, q1, h_l ,total_lossd= model(x, adj)

        if epoch % 1 == 0:
            tmp_q = q.data
            p = target_distribution(tmp_q)
            p1 = target_distribution(q1.data)

        ae_loss = F.mse_loss(x_bar, x)
        loss_a = F.mse_loss(adj_hat, adj.to_dense())
        re_loss = 1 * ae_loss+loss_a

        klh_loss = F.kl_div(q.log(), p, reduction='batchmean')
        q1q_loss = F.kl_div(q.log(), q1, reduction='batchmean')
        klz_loss = F.kl_div(q1.log(), p1, reduction='batchmean')
        loss = 1 * re_loss+klz_loss+args.alpha * klh_loss+args.beta*(total_lossd+q1q_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        res = q1.data.cpu().numpy().argmax(1)
        acc, nmi, ari = eva(y, res, str(epoch) + 'Q')
        logger.info("epoch%d%s:\t%s" % (epoch, ' Q', record_info([ nmi, ari])))
        loss_list.append(loss.item())
        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        if acc >= original_acc:
            original_acc = acc
            torch.save(model.state_dict(), './model_save/{}.pkl'.format(args.name))
    best_nmi = max(nmi_result)
    t_ari = ari_result[np.where(nmi_result == np.max(nmi_result))[0][0]]
    t_epoch = np.where(nmi_result == np.max(nmi_result))[0][0]
    logger.info("%sepoch%d:\t%s" % ('Best nmi is at ', t_epoch, record_info([ best_nmi, t_ari])))


if __name__ == "__main__":

    setup_seed(2018)
    data_path = 'data/' + args.name + '/data.tsv'
    device = torch.device("cuda")

    adj, x, y, cells, _ = load_data(data_path, args.name, args.n_input, True, args.n_clusters)
    adj = torch.tensor(adj, dtype=torch.float32).to(device)
    dataset = LoadDataset(x)
    x = torch.Tensor(dataset.x).to(device)
    model = DGCSG(
        ae_n_enc_1=500,
        ae_n_enc_2=2000,
        ae_n_dec_1=2000,
        ae_n_dec_2=500,
        hidden_size1=500,
        hidden_size2=2000,
        alpha=0.3,
        n_input= args.n_input,
        n_z= args.n_z,
        n_clusters= args.n_clusters,
        v=1.0).to(device)

    train(model, x, y)
