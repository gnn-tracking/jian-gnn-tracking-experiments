import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor

from torch_cluster import knn

class MHGravNetConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, space_dimensions: int, propagate_dimensions: int, k: int, num_heads: int, **kwargs):
        super().__init__(aggr=['mean', 'max'], flow='source_to_target', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.space_dimensions = space_dimensions
        self.propogate_dimensions = propagate_dimensions
        self.k = k
        self.num_heads = num_heads

        self.layernorm1 = torch.nn.LayerNorm(out_channels)
        self.layernorm2 = torch.nn.LayerNorm(out_channels)
        self.layernorm3 = torch.nn.LayerNorm(out_channels)

        self.lin_embed = Linear(in_channels, num_heads * space_dimensions)
        self.lin_features = Linear(in_channels, num_heads * propagate_dimensions)

        self.lin_out1 = Linear(in_channels, out_channels, bias=False)
        self.lin_out2 = Linear(2 * num_heads * propagate_dimensions, out_channels)

        self.lin_mes1 = Linear(2 * propagate_dimensions, 3 * propagate_dimensions)
        self.lin_mes2 = Linear(3 * propagate_dimensions, propagate_dimensions)

        self.act = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(0.2)

        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters()
        self.lin_embed.reset_parameters()
        self.lin_features.reset_parameters()
        self.lin_out1.reset_parameters()
        self.lin_out2.reset_parameters()


    def forward(self, x: Tensor, drop=False) -> Tensor:
        # type: (Tensor, OptTensor) -> Tensor  # noqa
        # type: (PairTensor, Optional[PairTensor]) -> Tensor  # noqa

        h_l: Tensor = self.lin_features(x).view(-1, self.propogate_dimensions)
        s_l: Tensor = self.lin_embed(x).view(-1, self.space_dimensions)
        edge_indecies = torch.arange(1, self.num_heads + 1).repeat_interleave(x.size()[0]).to('cuda')

        edge_index = knn(s_l, s_l, self.k, edge_indecies, edge_indecies).flip([0])

        edge_weight = (s_l[edge_index[0]] - s_l[edge_index[1]]).pow(2).sum(-1)
        edge_weight = torch.exp(-10. * edge_weight)  # 10 gives a better spread

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=(h_l, h_l), edge_weight=edge_weight, size=(s_l.size(0), s_l.size(0))).view(x.size()[0], -1)

        if drop:
            out = self.dropout(out)

        return self.layernorm2(self.lin_out1(x) + self.lin_out2(out))

    def message(self, x_j: Tensor, x_i: Tensor, edge_weight: Tensor) -> Tensor:
        mes = self.lin_mes2(self.act(self.lin_mes1(torch.cat([x_i, x_j], dim=-1))))
        return (mes/torch.linalg.vector_norm(mes, dim=-1).unsqueeze(-1)) * edge_weight.unsqueeze(1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, k={self.k})')