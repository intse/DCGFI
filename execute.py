import torch
import numpy as np
import random
from Microservices.FTP.DCGFI.model.Trianer import DCGFITrainer
from Microservices.FTP.DCGFI.model.CGCN import CGCN

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    dataset='SN'
    num_class = 4
    num_node_type = 4
    num_edge_types = 11
    node_dim = 303
    batch_size = 128
    n_epochs = 100
    train_ratio = 0.8
    for i in range(5):
        setup_seed(42+i)
        out_channels = 32
        num_layer = 2
        k = 10
        beta = 1
        lr = 0.001
        lrstr = '0.001'

        model_name = "./Experiments/"+dataset+"/best_network.pth"
        model_path = "./Experiments/"+dataset+"/"

        TrainerI = DCGFITrainer(batch_size, lr, n_epochs, model_path, dataset,out_channels,num_class,beta,k,train_ratio)
        net = CGCN(num_layer, node_dim, out_channels, num_node_type,num_edge_types,num_class)

        net = TrainerI.train(net)

        net.load_state_dict(torch.load(model_name))
        # TrainerI.validate(net)
        TrainerI.test(net)


