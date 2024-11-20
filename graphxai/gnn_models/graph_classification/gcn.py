import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv

class GCN_2layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_2layer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        x = global_mean_pool(x, batch)

        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

class GCN_3layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_3layer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index,batch =1,edge_weights = None):
        if edge_weights is None:
            
            x = self.conv1(x, edge_index)
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            x = self.conv2(x, edge_index)
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            x = self.conv3(x, edge_index)
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
            #x = x.relu()
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
        else:
            
            x = self.conv1(x, edge_index,edge_weight=edge_weights)
            
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            x = self.conv2(x, edge_index,edge_weight=edge_weights)
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            x = self.conv3(x, edge_index,edge_weight=edge_weights)
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
            #x = x.relu()
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
        x = global_mean_pool(x, batch)

        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

    def embedding(self, x, edge_index,batch =1,edge_weights = None):
        if edge_weights is None:
            
            x = self.conv1(x, edge_index)
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            x = self.conv2(x, edge_index)
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            x = self.conv3(x, edge_index)
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
            #x = x.relu()
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
        else:
            
            x = self.conv1(x, edge_index,edge_weight=edge_weights)
            
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            x = self.conv2(x, edge_index,edge_weight=edge_weights)
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
            x = x.relu()
            x = self.conv3(x, edge_index,edge_weight=edge_weights)
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
            #x = x.relu()
            #x = torch.nn.functional.normalize(x, p=2, dim=1)
        x = global_mean_pool(x, batch)

        #x = F.dropout(x, p=0.5, training=self.training)
        

        return x

