import torch

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

from torch_cluster import knn
from pytorch_lightning.core.mixins import HyperparametersMixin

class GravNet(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, space_dimensions: int, k: int = 4, message_multiple: int = 2, **kwargs):
        super().__init__(aggr=['mean', 'max'], flow='source_to_target', **kwargs)

        assert not (in_channels != out_channels and message_multiple == 0)

        self.lin_embed = Linear(in_channels, 2 * in_channels + space_dimensions)
        self.lin_out = Linear(2 * out_channels + in_channels, out_channels, bias=False)

        if round(message_multiple * in_channels) > 0:
            self.lin_message = torch.nn.Sequential(
                Linear(2 * in_channels, round(message_multiple * in_channels)),
                torch.nn.LeakyReLU(),
                Linear(round(message_multiple * in_channels), out_channels),
            )

        # self.norm = torch.nn.LayerNorm(out_channels)

        self.in_channels = in_channels
        self.k = k

    def forward(self, x, batch_index = None):

        m_1, m_2, s = self.lin_embed(x).split(self.in_channels, dim=-1)
        edge_index = knn(s, s, self.k, batch_index, batch_index).flip([0])
        edge_weight = torch.exp(-10. * (s[edge_index[0]] - s[edge_index[1]]).pow(2).sum(-1))
        out = self.propagate(edge_index, x=(m_1, m_2), edge_weight=edge_weight, size=None).view(x.size()[0], -1)

        return self.lin_out(torch.cat([x, out], dim=-1))

    def message(self, x_i, x_j, edge_weight):
        if self.lin_message != None:
            mes = self.lin_message(torch.cat([x_j, x_i], dim=-1))
            return (mes/torch.linalg.vector_norm(mes, dim=-1).unsqueeze(-1)) * edge_weight.unsqueeze(1)
        else:
            return x_j * edge_weight.unsqueeze(1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, k={self.k})')
    
class Model(torch.nn.Module, HyperparametersMixin):
    def __init__(self, embed_dim, space_dim, num_layers, k = 4, message_multiple = 2, input_dim = 14, output_dim = 4):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GravNet(in_channels=embed_dim, out_channels=embed_dim, space_dimensions=space_dim, k=k, message_multiple=message_multiple))
        self.linear_in = Linear(input_dim, embed_dim)
        self.linear_out = Linear(embed_dim, output_dim + 1)

        self.act = torch.nn.LeakyReLU()
    
    def forward(self, batch, batch_index):
        batch = self.act(self.linear_in(batch))
        for layer in self.layers():
            batch = self.act(layer(batch, batch_index))
        batch = self.linear_out(batch)

        return {
            "B" : torch.sigmoid(batch[:, 0]),
            "H" : batch[:, 1:]
        }
