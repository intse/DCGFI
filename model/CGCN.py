import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from Microservices.FTP.DCGFI.model.CRConv import CRConv
from torch_geometric.nn.dense.linear import HeteroLinear

class CGCN(nn.Module):
    def __init__(self,num_layers,node_dim,out_channels,num_node_types,num_edge_types,num_class):
        super(CGCN, self).__init__()
        self.hetero_lin = HeteroLinear(node_dim, out_channels,num_node_types, bias=True)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict= nn.ModuleDict()
            for rel_type in range(num_edge_types):
                conv_dict[str(rel_type)]=CRConv(in_channels=out_channels,
                                                out_channels=out_channels,
                                                num_edge_types=num_edge_types)
            self.convs.append(conv_dict)
        self.fuse = nn.Sequential(
            nn.Linear(out_channels * num_edge_types, out_channels),
            nn.ReLU()
        )
        self.classifier=nn.Linear(out_channels,num_class)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data):
        x,node_type, edge_index,edge_type=data.x,data.node_type,data.edge_index,data.edge_type
        x = self.hetero_lin(x,node_type)
        for layer in self.convs:
            layer_out = []
            for rel_type, conv in layer.items():
                mask = (edge_type == int(rel_type))
                edge_index_rel = edge_index[:, mask]
                edge_type_rel = edge_type[mask]
                out = conv(x, edge_index_rel,edge_type_rel)
                layer_out.append(out)
            x = self.fuse(torch.cat(layer_out, dim=1))
        graph_embeddings = global_mean_pool(x,data.batch)
        graph_outputs = self.classifier(graph_embeddings)
        return graph_embeddings,graph_outputs