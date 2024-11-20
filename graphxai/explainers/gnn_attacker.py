import torch
#import ipdb

from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

from graphxai.explainers._base import _BaseExplainer
from graphxai.utils import Explanation, node_mask_from_edge_mask

import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

EPS = 1e-15

class GNNAttacker(_BaseExplainer):
    """
    GNNExplainer: node only
    """
    def __init__(self, model: torch.nn.Module, coeff: dict = None):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
                The output of the model should be unnormalized class score.
                For example, last layer = CNConv or Linear.
            coeff (dict, optional): coefficient of the entropy term and the size term
                for learning edge mask and node feature mask
                Default setting:
                    coeff = {'edge': {'entropy': 1.0, 'size': 0.005},
                             'feature': {'entropy': 0.1, 'size': 1.0}}
        """
        super().__init__(model)
        if coeff is not None:
            self.coeff = coeff
        else:
            self.coeff = {'edge': {'entropy': 1.0, 'size': 0.005},
                          'feature': {'entropy': 0.1, 'size': 1.0}}

    def get_explanation_node(self, 
            node_idx: int, 
            x: torch.Tensor,                 
            edge_index: torch.Tensor,
            label: torch.Tensor = None,
            num_hops: int = None,
            explain_feature: bool = True,
            y = None,
            forward_kwargs: dict = {}):
        """
        Explain a node prediction.

        Args:
            node_idx (int): index of the node to be explained
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            x (torch.Tensor, [n x d]): node features
            label (torch.Tensor, optional, [n x ...]): labels to explain
                If not provided, we use the output of the model.
            num_hops (int, optional): number of hops to consider
                If not provided, we use the number of graph layers of the GNN.
            explain_feature (bool): whether to compute the feature mask or not
                (:default: :obj:`True`)
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index

        Returns:
            exp (dict):
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor): k-hop edge importance
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved
        """
        label = self._predict(x.to(device), edge_index.to(device),
                              forward_kwargs=forward_kwargs)# if label is None else label
        num_hops = self.L if num_hops is None else num_hops

        org_eidx = edge_index.clone().to(device)

        khop_info = subset, sub_edge_index, mapping, hard_edge_mask = \
            k_hop_subgraph(node_idx, num_hops, edge_index,
                           relabel_nodes=True) #num_nodes=x.shape[0])
        sub_x = x[subset].to(device)

        # print('sub_x shape', sub_x.shape)
        # print('sub edge index shape', sub_edge_index.shape)

        self._set_masks(sub_x.to(device), sub_edge_index.to(device), explain_feature=explain_feature)

        self.model.eval()
        num_epochs = 200

        # Loss function for GNNExplainer's objective
        def loss_fn(log_prob, mask, mask_type):
            # Select the log prob and the label of node_idx
            node_log_prob = log_prob[torch.where(subset==node_idx)].squeeze()
            node_label = label[mapping]
            # Maximize the probability of predicting the label (cross entropy)
            loss = -node_log_prob[node_label].item()
            a = mask.sigmoid()
            # Size regularization
            loss += self.coeff[mask_type]['size'] * torch.sum(a)
            # Element-wise entropy regularization
            # Low entropy implies the mask is close to binary
            entropy = -a * torch.log(a + 1e-15) - (1-a) * torch.log(1-a + 1e-15)
            loss += self.coeff[mask_type]['entropy'] * entropy.mean()
            return loss

        def train(mask, mask_type):
            optimizer = torch.optim.Adam([mask], lr=0.01)
            for epoch in range(1, num_epochs+1):
                optimizer.zero_grad()
                if mask_type == 'feature':
                    h = sub_x.to(device) * mask.view(1, -1).sigmoid().to(device)
                else:
                    h = sub_x.to(device)
                log_prob = self._predict(h.to(device), sub_edge_index.to(device), return_type='log_prob')
                loss = loss_fn(log_prob, mask, mask_type)
                loss.backward()
                optimizer.step()

        feat_imp = None
        if explain_feature: # Get a feature mask
            train(self.feature_mask, 'feature')
            feat_imp = self.feature_mask.data.sigmoid()

        train(self.edge_mask, 'edge')
        edge_imp = self.edge_mask.data.sigmoid().to(device)
        for i in range(0,len(edge_imp),2):
            edge_imp[i] = (edge_imp[i]+edge_imp[i+1])/2
            edge_imp[i+1] = edge_imp[i]
        # print('pre activation edge_imp:', edge_imp)

        # print('IN GNNEXPLAINER')
        # print('edge imp shape', edge_imp.shape)

        self._clear_masks()

        discrete_edge_mask = (edge_imp > 0.5) # Turn into bool activation because of sigmoid

        khop_info = (subset, org_eidx[:,hard_edge_mask], mapping, hard_edge_mask)

        exp = Explanation(
            feature_imp = feat_imp,
            node_imp = node_mask_from_edge_mask(khop_info[0], khop_info[1],discrete_edge_mask),
            #edge_imp = discrete_edge_mask.float(),
            edge_imp = edge_imp.float(),
            node_idx = node_idx
        )

        exp.set_enclosing_subgraph(khop_info)

        return exp

    # def get_explanation_graph(self, x: torch.Tensor, edge_index: torch.Tensor,
    #                           label: torch.Tensor = None,
    #                           forward_kwargs: dict = None, explain_feature: bool = False):
    #     """
    #     Explain a whole-graph prediction.

    #     Args:
    #         edge_index (torch.Tensor, [2 x m]): edge index of the graph
    #         x (torch.Tensor, [n x d]): node features
    #         label (torch.Tensor, [n x ...]): labels to explain
    #         forward_kwargs (dict, optional): additional arguments to model.forward
    #             beyond x and edge_index
    #         explain_feature (bool): whether to explain the feature or not

    #     Returns:
    #         exp (dict):
    #             exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
    #             exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
    #             exp['node_imp'] (torch.Tensor, [m]): k-hop node importance
    #     """
    #     raise Exception('GNNExplainer does not support graph-level explanation.')

    def graph_hot_attack(self, x, edge_index,exp_id, label = None, forward_kwargs = {}, lr = 0.01,cand_len=10):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for a graph.

        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self._clear_masks()

        # all nodes belong to same graph
        # batch = torch.zeros(x.shape[0], dtype=int, device=x.device)

        # Get the initial prediction.
        with torch.no_grad():
            # out = self.model(x=x, edge_index=edge_index, **forward_kwargs)
            # # if self.return_type == 'regression':
            # #     prediction = out
            # # else:
            # log_logits = self.__to_log_prob__(out)
            
            log_logits = self._predict(x.to(device), edge_index.to(device), forward_kwargs = forward_kwargs, return_type='log_prob')
            pred_label = log_logits.argmax(dim=-1)
        if label is not None:
            pred_label = [label]
            
        (n, d), m = x.shape, edge_index.shape[1]
        std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * n))
        edge_mask = (torch.randn(m) * std).to(device)
        edge_mask[exp_id]=0.00001
        #print(edge_mask)
        self.edge_mask = torch.nn.Parameter(edge_mask)
        #self._set_masks(x, edge_index, edge_mask = None, explain_feature = True, device = x.device)
        #self.to(x.device)
        #if self.allow_edge_mask:
        # self.feature_mask = self.feature_mask.to(x.device)
        # self.edge_mask = self.edge_mask.to(x.device)
        parameters = [self.edge_mask]
        #else:
            #parameters = [self.feature_mask]
        #ipdb.set_trace()
        optimizer = torch.optim.Adam(parameters, lr=lr)

        # if self.log:  # pragma: no cover
        #     pbar = tqdm(total=self.epochs)
        #     pbar.set_description('Explain graph')

        def loss_fn(node_idx, log_logits, pred_label):
            # node_idx is -1 for explaining graphs
            # if self.return_type == 'regression':
            #     if node_idx != -1:
            #         loss = torch.cdist(log_logits[node_idx], pred_label[node_idx])
            #     else:
            #         loss = torch.cdist(log_logits, pred_label)
            # else:
            if node_idx != -1:
                loss = -log_logits[node_idx, pred_label[node_idx]]
            else:
                loss = -log_logits[0, pred_label[0]]

            m = self.edge_mask.sigmoid()
            #edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeff['edge']['size'] * torch.sum(m)
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            #loss = loss + self.coeffs['edge_ent'] * ent.mean()
            loss = loss + self.coeff['edge']['entropy'] * ent.mean()


            return loss
        fliter = torch.ones(self.edge_mask.shape).to(device)
        bias = torch.zeros(self.edge_mask.shape).to(device)
        fliter[exp_id]=0
        bias[exp_id]=0.999
        num_epochs = 10 # TODO: make more general
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()
            #h = x * self.feature_mask.sigmoid()
            # out = self.model(x=h, edge_index=edge_index, **forward_kwargs)

            # log_logits = self.__to_log_prob__(out)
            
            log_logits = self._predict(x.to(device), edge_index.to(device), edge_weights=self.edge_mask.to(device)*fliter+bias,forward_kwargs = forward_kwargs, return_type='log_prob')

            loss = loss_fn(-1, log_logits, pred_label)
            loss.backward()
            optimizer.step()
            #print(self.edge_mask)
        #     if self.log:  # pragma: no cover
        #         pbar.update(1)

        # if self.log:  # pragma: no cover
        #     pbar.close()

        edge_mask = self.edge_mask.detach().sigmoid().cpu()
        #print(edge_mask)
        edges = edge_index.transpose(1,0)
        #print(edges)
        #print("......")
        positives = np.argsort(np.array(-edge_mask))
        positive = []
        h = 0
        print(edges[exp_id])
        while len(positive)<cand_len:
            if h==len(positives):
                break
            if positives[h] in exp_id:
                h+=1
                continue
            positive.append(positives[h])
            h+=1
        print(edges[positive])
        return edges[positive]

    def graph_cold_attack(self, x, edge_index,exp_id, label = None, forward_kwargs = {}, lr = 0.01,cand_len=10):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for a graph.

        Args:
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self._clear_masks()

        # all nodes belong to same graph
        # batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
        r_edge_index = [edge_index[0,exp_id].tolist(),edge_index[1,exp_id].tolist()]
        
        adj = np.zeros((x.shape[0],x.shape[0]))
        for i in range(len(edge_index[0])):
            adj[edge_index[0,i],edge_index[1,i]]=1
        for i in range(x.shape[0]):
            for j in range(i,x.shape[0]):
                if adj[i,j]==0:
                    r_edge_index[0].append(i)
                    r_edge_index[1].append(j)
                    r_edge_index[0].append(j)
                    r_edge_index[1].append(i)
        r_edge_index= torch.tensor(r_edge_index)
        # Get the initial prediction.
        with torch.no_grad():
            # out = self.model(x=x, edge_index=edge_index, **forward_kwargs)
            # # if self.return_type == 'regression':
            # #     prediction = out
            # # else:
            # log_logits = self.__to_log_prob__(out)
            log_logits = self._predict(x.to(device), edge_index.to(device), forward_kwargs = forward_kwargs, return_type='log_prob')
            pred_label = log_logits.argmax(dim=-1)
        if label is not None:
            pred_label = [label]
            
        (n, d), m = x.shape, r_edge_index.shape[1]
        std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * n))
        edge_mask = (torch.randn(m) * std).to(device)
        edge_mask[0:len(exp_id)]=0.00001
        self.r_edge_mask = torch.nn.Parameter(edge_mask)
        #self._set_masks(x, edge_index, edge_mask = None, explain_feature = True, device = x.device)
        #self.to(x.device)
        #if self.allow_edge_mask:
        # self.feature_mask = self.feature_mask.to(x.device)
        # self.edge_mask = self.edge_mask.to(x.device)
        parameters = [self.r_edge_mask]
        #else:
            #parameters = [self.feature_mask]
        #ipdb.set_trace()
        optimizer = torch.optim.Adam(parameters, lr=lr)

        # if self.log:  # pragma: no cover
        #     pbar = tqdm(total=self.epochs)
        #     pbar.set_description('Explain graph')

        def loss_fn_2(node_idx, log_logits, pred_label):
            # node_idx is -1 for explaining graphs
            # if self.return_type == 'regression':
            #     if node_idx != -1:
            #         loss = torch.cdist(log_logits[node_idx], pred_label[node_idx])
            #     else:
            #         loss = torch.cdist(log_logits, pred_label)
            # else:
            if node_idx != -1:
                loss = -log_logits[node_idx, pred_label[node_idx]]
            else:
                loss = -log_logits[0, pred_label[0]]

            m = self.r_edge_mask.sigmoid()
            #edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
            loss = loss + self.coeff['edge']['size'] * torch.sum(m)
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            #loss = loss + self.coeffs['edge_ent'] * ent.mean()
            loss = loss + self.coeff['edge']['entropy'] * ent.mean()

            return -loss
        fliter = torch.ones(self.r_edge_mask.shape).to(device)
        bias = torch.zeros(self.r_edge_mask.shape).to(device)
        fliter[0:len(exp_id)]=0
        bias[0:len(exp_id)]=0.999
        num_epochs = 10 # TODO: make more general
        for epoch in range(1, num_epochs + 1):
            optimizer.zero_grad()
            #h = x * self.feature_mask.sigmoid()
            # out = self.model(x=h, edge_index=edge_index, **forward_kwargs)

            # log_logits = self.__to_log_prob__(out)
            
            log_logits = self._predict(x.to(device), r_edge_index.to(device), edge_weights=self.r_edge_mask.to(device)*fliter+bias,forward_kwargs = forward_kwargs, return_type='log_prob')

            loss = loss_fn_2(-1, log_logits, pred_label)
            loss.backward()
            optimizer.step()
            
        #     if self.log:  # pragma: no cover
        #         pbar.update(1)

        # if self.log:  # pragma: no cover
        #     pbar.close()

        edge_mask = self.r_edge_mask.detach().sigmoid().cpu()

        edges = r_edge_index.transpose(1,0)
        #print(edge_mask)
        #print(edges)
        #print("------")
        positives = np.argsort(np.array(-edge_mask))
        positive = []
        c = 0
        n_id = [t for t in range(len(exp_id))]
        while len(positive)<cand_len:
            if c==len(positives):
                break
            if positives[c] in n_id:
                c+=1
                continue
            positive.append(positives[c])
            c+=1
        print([edge_index[0,exp_id].tolist(),edge_index[1,exp_id].tolist()])
        print(edges[positive])
        return edges[positive]
    
    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()
