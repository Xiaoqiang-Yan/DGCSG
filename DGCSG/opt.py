import argparse

parser = argparse.ArgumentParser(description='DGCSG', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', type=str, default="Pollen")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--n_z', type=int, default=10)


args = parser.parse_args()
print("Network setting…")

if args.name == 'Biase':
    args.lr = 1e-4
    args.n_clusters = 3
    args.n_input = 2000
    args.cuda = False
    args.alpha = 10
    args.beta = 0.00003
elif args.name == 'Klein':
    args.lr = 1e-4
    args.n_clusters = 4
    args.n_input = 2000
    args.cuda = False
    args.alpha = 10
    args.beta = 0.00003
elif args.name == 'Chung':
    args.lr = 1e-4
    args.n_clusters = 5
    args.n_input = 2000
    args.cuda = False
    args.alpha = 10
    args.beta = 0.00003
elif args.name == 'Sun1':
    args.lr = 1e-4
    args.n_clusters = 6
    args.n_input = 512
    args.cuda = False
    args.alpha = 10
    args.beta = 0.00003
elif args.name == 'Sun2':
    args.lr = 1e-4
    args.n_clusters = 8
    args.n_input = 512
    args.cuda = False
    #args.alpha = 10
    args.alpha = 10
    #args.beta = 0.00003
    args.beta = 0.00003
elif args.name == 'Sun3':
    args.lr =1e-4
    args.n_clusters = 7
    args.n_input = 512
    args.cuda = False
    args.alpha = 10
    args.beta = 0.00003
elif args.name == 'Deng':
    args.lr =1e-4
    args.n_clusters = 6
    args.n_input = 2000
    args.cuda = False
    args.alpha = 10
    args.beta = 0.00003
elif args.name == 'Muraro':
    args.lr = 1e-4
    args.n_clusters = 9
    args.n_input = 2000
    args.cuda = False
    #args.alpha = 10
    args.alpha = 10
    #args.beta = 0.00003
    args.beta = 0.00003
elif args.name == 'Pollen':
    args.lr = 1e-4
    args.n_clusters = 11
    args.n_input = 2000
    args.cuda = False
    #args.alpha = 10
    args.alpha = 10
    #args.beta = 0.00003
    args.beta = 0.00003
elif args.name == 'Darmanis':
    args.lr = 1e-4
    args.n_clusters = 9
    args.n_input = 2000
    args.cuda = False
    args.alpha = 10
    args.beta = 0.00003
elif args.name == 'bladder':
    args.lr = 1e-5
    args.n_clusters = 16
    args.n_input = 2000
    args.cuda = False
    args.alpha = 10
    args.beta = 0.00003
elif args.name == '10X_PBMC':
    args.lr = 1e-6
    args.n_clusters = 8
    args.n_input = 2000
    args.cuda = False
    args.alpha = 10
    args.beta = 0.00003
elif args.name == 'Habib':
    args.lr = 1e-4
    args.n_clusters = 10
    args.n_input = 2000
    args.cuda = False
    args.alpha = 10
    args.beta = 0.00003
elif args.name == 'Zeisel':
    args.lr = 1e-4
    args.n_clusters = 9
    args.n_input = 2000
    args.cuda = False
    args.alpha = 10
    args.beta = 0.00003
else:
    print("error!")
