import numpy as np
import torch
import torch.nn as nn
import tqdm
import time
from collections import OrderedDict

from typing import Optional
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv

from graphxai.explainers._base import _BaseExplainer
from graphxai.utils import Explanation, node_mask_from_edge_mask

from graphxai.gnn_models.graph_classification.gin import GINWConv
device = "cuda" if torch.cuda.is_available() else "cpu"

from torch.nn import functional as F
from torch.nn import ModuleList, Linear as Lin
from torch_geometric.nn import BatchNorm, ARMAConv


class MLP(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, act=nn.Tanh()):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(OrderedDict([
                ('lin1', Lin(in_channels, hidden_channels)),
                ('act', act),
                ('lin2', Lin(hidden_channels, out_channels))
                ]))
     
    def forward(self, x):
        return self.mlp(x)

class EdgeMaskNet(torch.nn.Module):

    def __init__(self,
                 n_in_channels,
                 e_in_channels,
                 hid=72, n_layers=3):
        super(EdgeMaskNet, self).__init__()

        self.node_lin = Lin(n_in_channels, hid)
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(n_layers):
            conv = ARMAConv(in_channels=hid, out_channels=hid)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hid))

        if e_in_channels > 1:
            self.edge_lin1 = Lin(2 * hid, hid)
            self.edge_lin2 = Lin(e_in_channels, hid)
            self.mlp = MLP(2 * hid, hid, 1)
        else:
            self.mlp = MLP(2 * hid, hid, 1)
        self._initialize_weights()
        
    def forward(self, x, edge_index, edge_attr=None):

        x = torch.flatten(x, 1, -1)
        x = F.relu(self.node_lin(x))
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x, edge_index))
            x = batch_norm(x)

        e = torch.cat([x[edge_index[0, :]], x[edge_index[1, :]]], dim=1)
        if not edge_attr is None:
            if edge_attr.size(-1) > 1:
                e1 = self.edge_lin1(e)
                e2 = self.edge_lin2(edge_attr)
                e = torch.cat([e1, e2], dim=1)  # connection

        return self.mlp(e)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight) 


class ReFine(_BaseExplainer):
    """
    PGExplainer

    Code adapted from https://github.com/Wuyxin/ReFine
    """
    def __init__(self, model: nn.Module, emb_layer_name: str = None,
                 explain_graph: bool = False,
                 coeff_size: float = 1e-4, coeff_ent: float = 1e-2,
                 t0: float = 5.0, t1: float = 2.0, gamma=1,
                 lr: float = 0.003, max_epochs: int = 20, eps: float = 1e-3,
                 hidden: int= 64, n_layers: int = 2, num_hops: int = None, in_channels = None, batch_size=64):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
                The output of the model should be unnormalized class score.
                For example, last layer = CNConv or Linear.
            emb_layer_name (str, optional): name of the embedding layer
                If not specified, use the last but one layer by default.
            explain_graph (bool): whether the explanation is graph-level or node-level
            coeff_size (float): size regularization to constrain the explanation size
            coeff_ent (float): entropy regularization to constrain the connectivity of explanation
            t0 (float): the temperature at the first epoch
            t1 (float): the temperature at the final epoch
            lr (float): learning rate to train the explanation network
            max_epochs (int): number of epochs to train the explanation network
            num_hops (int): number of hops to consider for node-level explanation
        """
        super().__init__(model, emb_layer_name)

        # Parameters for PGExplainer
        self.explain_graph = explain_graph
        self.coeff_size = coeff_size
        self.coeff_ent = coeff_ent
        self.t0 = t0
        self.t1 = t1
        self.lr = lr
        self.eps = eps
        self.max_epochs = max_epochs
        self.num_hops = self.L if num_hops is None else num_hops
        self.batch_size=batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Explanation model in PGExplainer

        mult = 2 # if self.explain_graph else 3
        """
        if in_channels is None:
            if isinstance(self.emb_layer, GCNConv):
                in_channels = mult * self.emb_layer.out_channels
            elif isinstance(self.emb_layer, GINConv):
                in_channels = mult * self.emb_layer.nn.out_features
            elif isinstance(self.emb_layer, GINWConv):
                in_channels = mult * self.emb_layer.nn.out_features
            elif isinstance(self.emb_layer, torch.nn.Linear):
                in_channels = mult * self.emb_layer.out_features
            else:
                fmt_string = 'PGExplainer not implemented for embedding layer of type {}, please provide in_channels directly.'
                raise NotImplementedError(fmt_string.format(type(self.emb_layer)))
        """
        self.elayers = EdgeMaskNet(
            n_in_channels = in_channels,
            e_in_channels = 0, 
            hid = hidden,
            n_layers = n_layers,
            ).to(device)
        self.gamma = gamma


    def __concrete_sample(self, log_alpha: torch.Tensor,
                          beta: float = 1.0, training: bool = True):
        """
        Sample from the instantiation of concrete distribution when training.

        Returns:
            training == True: sigmoid((log_alpha + noise) / beta)
            training == False: sigmoid(log_alpha)
        """
        if training:
            random_noise = torch.rand(log_alpha.shape).to(device)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def _reparameterize(self, log_alpha, beta=1, training=True):
        
        EPS = 1e-6
        if training:
            random_noise = torch.rand(log_alpha.size()).to(self.device)
            gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta + EPS
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()
            
        return gate_inputs
 
       
       

    def __get_mask(self, x: torch.Tensor, edge_index: torch.Tensor,
                           node_idx: int = None,
                           forward_kwargs: dict = {},
                           tmp: float = 1.0, training: bool = False):
        """
        Compute the edge mask based on embedding.

        Returns:
            prob_with_mask (torch.Tensor): the predicted probability with edge_mask
            edge_mask (torch.Tensor): the mask for graph edges with values in [0, 1]
        """
#        if not self.explain_graph and node_idx is None:
#            raise ValueError('node_idx should be provided.')



        with torch.set_grad_enabled(training):
            # Concat relevant node embeddings
            # import ipdb; ipdb.set_trace()

            edge_weights = self.elayers(x, edge_index).view(-1)
            edge_weights = self._reparameterize(edge_weights)
            #print(edge_mask.shape)
            #print(edge_index.shape)
            
            #self.aggreate(V=int(max(edge_index[0])),edges = edge_index.transpose(1,0), imp = edge_weights)
            #"""
            n = x.shape[0]  # number of nodes
            mask_sparse = torch.sparse_coo_tensor(
                edge_index, edge_weights, (n, n))
            # Not scalable
            
            mask_sigmoid = mask_sparse.to_dense()
            # Undirected graph
            sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
            edge_mask = sym_mask[edge_index[0], edge_index[1]]
            #"""
            #self._set_masks(x, edge_index, edge_mask)

        # Compute the model prediction with edge mask
        # with torch.no_grad():
        #     tester = self.model(x, edge_index)
        #     print(tester)
        #self._clear_masks()

        return edge_mask,mask_sigmoid


    def fidelity_loss(self, log_logits, mask, pred_label):
        EPS = 1e-6
        idx = [i for i in range(len(pred_label))]
        loss = -torch.log(log_logits.softmax(dim=1)[idx, pred_label.view(-1)]).sum()
        #print(log_logits[0,pred_label] )
        #loss = -torch.log(log_logits[0,pred_label] + 1e-6)
        #print(loss)
        loss = loss + self.coeff_size * torch.sum(mask)#.mean()
        ent = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        loss = loss + self.coeff_ent * ent.mean()
        return loss
    
  
    def get_contrastive_loss(self, c, y, tau=0.1):
        
        
        c = c / c.norm(dim=1, keepdim=True)
        mat = F.relu(torch.mm(c, c.T))
        ttl_scores = torch.sum(mat, dim=1)
        
        pos_scores = torch.tensor([mat[i, y == y[i]].sum() for i in range(mat.shape[0])]).to(c.device)
        
        # contrastive_loss = - torch.log(torch.sum(pos_scores / ttl_scores, dim=0))
        contrastive_loss = - torch.logsumexp(pos_scores / (tau * ttl_scores), dim=0)
        
        return contrastive_loss
    
    def train_explanation_model(self, dataset: Data, forward_kwargs: dict = {}):
        """
        Train the explanation model.
        """
        optimizer = torch.optim.Adam(self.elayers.parameters(), lr=self.lr)

        def loss_fn(prob: torch.Tensor, ori_pred: int):
            # Maximize the probability of predicting the label (cross entropy)
            loss = -torch.log(prob[ori_pred] + 1e-6)
            # Size regularization
            print(loss)
            edge_mask = self.mask_sigmoid
            loss += self.coeff_size * torch.sum(edge_mask)
            # Element-wise entropy regularization
            # Low entropy implies the mask is close to binary
            edge_mask = edge_mask * 0.99 + 0.005
            entropy = - edge_mask * torch.log(edge_mask) \
                - (1 - edge_mask) * torch.log(1 - edge_mask)
            loss += self.coeff_ent * torch.mean(entropy)
            
            return loss

        if self.explain_graph:  # Explain graph-level predictions of multiple graphs
            # Get the embeddings and predicted labels
            with torch.no_grad():
                dataset_indices = list(range(len(dataset)))
                self.model.eval()
                emb_dict = {}
                ori_pred_dict = {}
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid].to(device)
                    pred_label = self._predict(data.x, data.edge_index,
                                               forward_kwargs=forward_kwargs)
                    emb = self._get_embedding(data.x, data.edge_index,
                                              forward_kwargs=forward_kwargs)
                    # OWEN inserting:
                    emb_dict[gid] = emb.to(device) # Add embedding to embedding dictionary
                    ori_pred_dict[gid] = pred_label

            # Train the mask generator
            duration = 0.0
            last_loss = 0.0
            for epoch in range(self.max_epochs):
                loss = 0.0
                loss_tmp = torch.zeros(1)
                pred_list = []
                embs=[]
                edge_masks =[]
                #mask_sigmoids =[]
                log_logits = []
                ys=[]
                tmp = float(self.t0 * np.power(self.t1/self.t0, epoch/self.max_epochs))
                self.elayers.train()
                optimizer.zero_grad()
                tic = time.perf_counter()
                batch=0
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid].to(device)
                    edge_mask, mask_sigmoid= self.__get_mask(
                        data.x, data.edge_index,
                        forward_kwargs=forward_kwargs,
                        tmp=tmp, training=True)
                    edge_masks.append(mask_sigmoid.to(device))
                    #mask_sigmoids.append(mask_sigmoid)
                    embs.append(self.model.to(device).embedding(data.x, data.edge_index,edge_weights=edge_mask.to(device),**forward_kwargs))
                    log_logits.append(self.model.to(device)(data.x, data.edge_index, edge_weights=edge_mask.to(device),**forward_kwargs))
                    #print(prob_with_mask,ori_pred_dict[gid])
                    #print(log_logits)
                    #print(edge_masks)
                    ys.append(data.y.to(device))
                    batch+=1 
                    if batch ==self.batch_size or gid== len(dataset)-1:
                        fid_loss = torch.zeros(1).to(device)
                        for i in range(batch):
                            
                            fid_loss += self.fidelity_loss(log_logits[i], edge_masks[i], ys[i])
                            pred_label = log_logits[i].argmax(-1).item()
                            pred_list.append(pred_label)
                        cts_loss = self.get_contrastive_loss(torch.cat(embs,dim=0),pred_list,tau=0.1)
                        loss_tmp=fid_loss+cts_loss
                        loss+=loss_tmp.item()
                        #print(cts_loss)
                        #print(fid_loss)
                        loss_tmp.backward()
                        batch=0
                        embs=[]
                        edge_masks=[]
                        log_logits = []
                        ys=[]
                        pred_list=[]
                #loss+=cts_loss
                #loss.backward()
                optimizer.step()
                duration += time.perf_counter() - tic
                print(f'Epoch: {epoch} | Loss: {loss}')
                if abs(last_loss - loss) < self.eps:
                    break
                last_loss = loss


            print(f"training time is {duration:.5}s")

    def get_explanation_graph(self, x: torch.Tensor, edge_index: torch.Tensor,
                              label: Optional[torch.Tensor] = None,
                              y = None,
                              forward_kwargs: dict = {},
                              top_k: int = 10):
        """
        Explain a whole-graph prediction.

        Args:
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            label (torch.Tensor, optional, [n x ...]): labels to explain
                If not provided, we use the output of the model.
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index
            top_k (int): number of edges to include in the edge-importance explanation

        Returns:
            exp (dict):
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
                exp['node_imp'] (torch.Tensor, [m]): k-hop node importance

        """
        if not self.explain_graph:
            raise Exception('For node-level explanations use `get_explanation_node`.')

        label = self._predict(x, edge_index,
                              forward_kwargs=forward_kwargs) if label is None else label

        with torch.no_grad():
            edge_mask,_ = self.__get_mask(
                        x, edge_index,
                        forward_kwargs=forward_kwargs, training=False)

        #exp['edge_imp'] = edge_mask

        exp = Explanation(
            node_imp = node_mask_from_edge_mask(torch.arange(x.shape[0], device=x.device), edge_index),
            edge_imp = edge_mask
        )

        exp.set_whole_graph(Data(x=x, edge_index=edge_index))

        return exp

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()
