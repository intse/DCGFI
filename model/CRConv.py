from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax

class CRConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, num_edge_types: int,
                 heads: int = 1, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.src_lin = Linear(in_channels,out_channels)
        self.dst_lin = Linear(in_channels,out_channels)
        self.rel_lin = Linear(num_edge_types,out_channels)
        self.att_lin = Linear(3 * out_channels, self.heads, bias=False)
        self.msg_lin = Linear(2*out_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.src_lin.reset_parameters()
        self.dst_lin.reset_parameters()
        self.rel_lin.reset_parameters()
        self.att_lin.reset_parameters()
        self.msg_lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_type: Tensor) -> Tensor:
        edge_type_rep = F.one_hot(edge_type,num_classes=11).float()
        out = self.propagate(edge_index, x=x, edge_type=edge_type_rep, size=None)
        out += x.view(-1, 1, self.out_channels)
        out = out.view(-1, self.heads * self.out_channels)
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_type: Tensor, index: Tensor, ptr: OptTensor,size_i: Optional[int]) -> Tensor:
        edge_type = self.rel_lin(edge_type)
        x_i = self.src_lin(x_i)
        x_j = self.dst_lin(x_j)
        alpha = torch.cat([x_i, x_j, edge_type], dim=-1)
        alpha = F.leaky_relu(self.att_lin(alpha))
        alpha = softmax(alpha, index, ptr, size_i)
        out = self.msg_lin(torch.cat([x_j,edge_type], dim=-1)).unsqueeze(-2)
        return out * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
