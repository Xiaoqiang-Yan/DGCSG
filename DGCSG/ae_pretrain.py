import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.cluster import KMeans
from AE import AE
from utils import LoadDataset, eva, load_data
from opt import args


def pretrain_ae(model, dataset, y, data_name):
    acc_reuslt = []
    nmi_result = []
    ari_result = []

    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-4)
    # 确保 ae_pretrain 目录存在
    save_dir = './ae_pretrain'
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(50):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()
            _, x_bar, _, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            z_ae, x_bar, _, _ = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=n_clusters, n_init=20,).fit(z_ae.data.cpu().numpy())
            acc, nmi, ari = eva(y, kmeans.labels_, epoch)
            acc_reuslt.append(acc)
            nmi_result.append(nmi)
            ari_result.append(ari)

    #torch.save(model.state_dict(), data_name+ '.pkl')
    # 保存模型到 ae_pretrain 目录
    save_path = os.path.join(save_dir, f'{data_name}.pkl')
    torch.save(model.state_dict(), save_path)


print(args)
n_clusters = args.n_clusters
n_input = args.n_input
n_z = args.n_z

model = AE(
    ae_n_enc_1=500,
    ae_n_enc_2=2000,
    ae_n_dec_1=2000,
    ae_n_dec_2=500,
    n_input=n_input,
    n_z=n_z).cuda()

data_path = 'data\{}\data.tsv'.format(args.name)
_,x,y,_,_ = load_data(data_path,args.name,args.n_input,True,args.n_clusters)
dataset = LoadDataset(x)
pretrain_ae(model, dataset, y, args.name)
