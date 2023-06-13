import warnings
from typing import Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor

from torch_cluster import knn

class MHGravNetConv(MessagePassing):
    r"""The GravNet operator from the `"Learning Representations of Irregular
    Particle-detector Geometry with Distance-weighted Graph
    Networks" <https://arxiv.org/abs/1902.07987>`_ paper, where the graph is
    dynamically constructed using nearest neighbors.
    The neighbors are constructed in a learnable low-dimensional projection of
    the feature space.
    A second projection of the input feature space is then propagated from the
    neighbors to each vertex using distance weights that are derived by
    applying a Gaussian function to the distances.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): The number of output channels.
        space_dimensions (int): The dimensionality of the space used to
           construct the neighbors; referred to as :math:`S` in the paper.
        propagate_dimensions (int): The number of features to be propagated
           between the vertices; referred to as :math:`F_{\textrm{LR}}` in the
           paper.
        k (int): The number of nearest neighbors.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{in}), (|\mathcal{V_t}|, F_{in}))`
          if bipartite,
          batch vector :math:`(|\mathcal{V}|)` or
          :math:`((|\mathcal{V}_s|), (|\mathcal{V}_t|))` if bipartite
          *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, in_channels: int, out_channels: int, space_dimensions: int, propagate_dimensions: int, k: int, num_heads: int, **kwargs):
        super().__init__(aggr=['mean', 'max'], flow='source_to_target', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.space_dimensions = space_dimensions
        self.propogate_dimensions = propagate_dimensions
        self.k = k
        self.num_heads = num_heads

        self.lin_embed = Linear(in_channels, num_heads * space_dimensions)
        self.lin_features = Linear(in_channels, num_heads * propagate_dimensions)

        self.lin_out1 = Linear(in_channels, out_channels, bias=False)
        self.lin_out2 = Linear(2 * num_heads * propagate_dimensions, out_channels)

        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters()
        self.lin_embed.reset_parameters()
        self.lin_features.reset_parameters()
        self.lin_out1.reset_parameters()
        self.lin_out2.reset_parameters()


    def forward(self, x: Tensor) -> Tensor:
        # type: (Tensor, OptTensor) -> Tensor  # noqa
        # type: (PairTensor, Optional[PairTensor]) -> Tensor  # noqa

        h_l: Tensor = self.lin_features(x).view(-1, self.propogate_dimensions)
        s_l: Tensor = self.lin_embed(x).view(-1, self.space_dimensions)
        edge_indecies = torch.arange(1, self.num_heads + 1).repeat_interleave(x.size()[0]).to('cuda')

        edge_index = knn(s_l, s_l, self.k, edge_indecies, edge_indecies).flip([0])

        edge_weight = (s_l[edge_index[0]] - s_l[edge_index[1]]).pow(2).sum(-1)
        edge_weight = torch.exp(-10. * edge_weight)  # 10 gives a better spread

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=(h_l, None), edge_weight=edge_weight, size=(s_l.size(0), s_l.size(0))).view(x.size()[0], -1)

        return self.lin_out1(x) + self.lin_out2(out)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return x_j * edge_weight.unsqueeze(1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, k={self.k})')