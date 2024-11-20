# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import sys
sys.path.append("models/")
#from mlp import MLP
import numpy as np
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split

import statistics
device = "cuda" if torch.cuda.is_available() else "cpu"
class HashAgent():
    def __init__(self,h="md5",T=30, p=0.5):
        '''
            h: the hash function in "md5","sha1","sha256"
            T: the subset amount
        '''

        super(HashAgent, self).__init__()
        self.T = T
        self.h= h 
        self.add_I = [ [] for _ in range(self.T)]
        
        for i in range(self.T):
            for j in range(self.T):
                if j==i:
                    continue
                if np.random.random()<=p:
                    self.add_I[i].append(j)
                                         
    def hash_edge(self,V, u,v):
        #"""
        hexstring = hex(V*u+v)
        hexstring= hexstring.encode()
        if self.h == "md5":
            hash_device = hashlib.md5()
        elif self.h == "sha1":
            hash_device = hashlib.sha1()
        elif self.h == "sha256":
            hash_device = hashlib.sha256()
        hash_device.update(hexstring)
        I = int(hash_device.hexdigest(),16)%self.T
        return I
    
    def generate_mixed_subgraphs(self, graph):
        
        mixed_subgraphs = []
        
        original = graph.edge_index
        nodes = range(graph.x.shape[0])

        V= graph.x.shape[0]
        
        
                    
        for i in range(self.T):
            mixed_subgraphs.append(Data(
                        x = graph.x,
                        y = graph.y,
                        edge_attr = graph.edge_attr,
                        edge_index = []
                    ))
            
        for i in range(len(original[0])):
            
            u=original[0,i]
            v=original[1,i]
            if u>v:
                I = self.hash_edge(V,v,u)
            else:
                I = self.hash_edge(V,u,v)
            mixed_subgraphs[I].edge_index.append([u,v])
            
        for i in range(V-1):
            for j in range(i+1,V):
                u=nodes[i]
                v=nodes[j]
                I = self.hash_edge(V,u,v)
                for k in range(len(self.add_I[I])):
                    if self.add_I[I][k]==I:
                        continue
                    mixed_subgraphs[self.add_I[I][k]].edge_index.append([u,v])
                    mixed_subgraphs[self.add_I[I][k]].edge_index.append([v,u])
                    
        deletes = []
        new_mixed_subgraphs = []
        for i in range(self.T):
            if len(mixed_subgraphs[i].edge_index)==0:
                continue
            mixed_subgraphs[i].edge_index = torch.tensor(mixed_subgraphs[i].edge_index,dtype=torch.int64).transpose(1,0)
            new_mixed_subgraphs.append(mixed_subgraphs[i])
            
        return new_mixed_subgraphs#mixed_subgraphs


def enlarge_graph(dataset,hasher):
    new_graphs = []
    new_exlanations = []
    grounds = []
    
    times=0
    stds=[]
    avg_e = []
    min_e =[]
    max_e =[]
    train_index= []
    val_index= []
    test_index= []
    for i in range(len(dataset.graphs)):
        start = len(new_graphs)
        graph = dataset.graphs[i]
        n = graph.x.shape[0]
        nodes = [v for v in range(n)]
        ground = torch.zeros((n,n)).to(dataset.device)
        for j in range(graph.edge_index[0].shape[0]):
            ground[graph.edge_index[0,j],graph.edge_index[1,j]]=1
        explanation = dataset.explanations[i]
        new_graphs.append(graph)
        new_exlanations.append(explanation)
        grounds.append(ground)
        mixed_graphs=hasher.generate_mixed_subgraphs(graph)
        
        if i %100 ==0:
            print(f"{i}/{len(dataset.graphs)}")
            
        edges = []
        new_graphs.extend(mixed_graphs)
        
        end = len(new_graphs)
        indexs = [j for j in range(start,end)]
        if i in dataset.train_index:
            train_index.extend(indexs)
        elif i in dataset.val_index:
            val_index.extend(indexs)
        elif i in dataset.test_index:
            test_index.extend(indexs)
            
        new_exlanations.extend([explanation for _ in range(len(mixed_graphs))])
        grounds.extend([ground for _ in range(len(mixed_graphs))])
        
    dataset.graphs = new_graphs
    dataset.explanations = new_exlanations
    dataset.ground = grounds
    
    dataset.train_index = torch.tensor(train_index)
    dataset.val_index = torch.tensor(val_index)
    dataset.test_index = torch.tensor(test_index)
    dataset.Y = torch.tensor([dataset.graphs[i].y for i in range(len(dataset.graphs))]).to(dataset.device)
    
    return  dataset
        
    
    
class RobustClassifier(nn.Module):
    def __init__(self,BaseClassifier,Hasher):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(RobustClassifier, self).__init__()
        self.BaseClassifier = BaseClassifier
        self.Hasher = Hasher
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
    def forward(self, graph):
        subgraphs = self.Hasher.generate_mixed_subgraphs(graph)
        
        for i in range(len(subgraphs)):
            subgraphs[i].exp_key = [i]
        loader = DataLoader(subgraphs, batch_size = len(subgraphs), shuffle = True)
        
        #x = torch.cat([subgraphs[i].x for i in range(len(subgraphs))]).to(self.device)
        #edge_index = torch.cat([subgraphs[i].edge_index for i in range(len(subgraphs))]).to(self.device)
        self.BaseClassifier.eval()
        outputs = []
        for i in range(len(subgraphs)):
            data=subgraphs[i]
            output = self.BaseClassifier(data.x,data.edge_index,batch =data.batch).cpu().detach()
            outputs.append(output)
        outputs = np.array(outputs)
        Y_labels = np.argmax(outputs,axis=1)
        vote_label = np.argmax(np.bincount(Y_labels))
        return vote_label
    def vote(self, graph):
        subgraphs = self.Hasher.generate_mixed_subgraphs(graph)
        
        for i in range(len(subgraphs)):
            subgraphs[i].exp_key = [i]
        loader = DataLoader(subgraphs, batch_size = len(subgraphs), shuffle = True)
        
        self.BaseClassifier.eval()
        outputs=[]
        Y_labels= []
        for i in range(len(subgraphs)):
            data=subgraphs[i].to(device)
            output = self.BaseClassifier(data.x,data.edge_index,batch =data.batch).cpu().detach()

            Y_labels.append(np.argmax(output,axis=1).item())
            
        count = np.bincount(Y_labels)
        votes = count.copy()
        vote_label = np.argmax(count)
        Yc = count[vote_label]
        count[vote_label]=-1
        second_label = np.argmax(count)
        Yb = count[second_label]
        
        if vote_label>second_label:
            Mc = (Yc-Yb-1)//2
        else:
            Mc = (Yc-Yb)//2
        return vote_label, Mc
    

class RobustExplainer():
    def __init__(self,BaseExplainer,BaseClassifier,Hasher,k=10,lamb=2,tau=0.2):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(RobustExplainer, self).__init__()
        self.BaseExplainer = BaseExplainer
        self.BaseClassifier = BaseClassifier
        self.Hasher = Hasher
        self.k = k
        self.tau = tau
        self.lamb = lamb
        
    def set_para(self,tau=0.3):
        self.tau = tau
         
        
        
    def explain(self, graph, label, k=10, lamb=2):
        self.k = k
        self.lamb = lamb
        
        mixed_graph = self.Hasher.generate_mixed_subgraphs(graph)
        
        loader = DataLoader(mixed_graph, batch_size = len(mixed_graph), shuffle = False)
        
        
        V= graph.x.shape[0]
        nodes = range(V)
        N=np.zeros((V+1,V+1))
        has_edge=np.zeros((V+1,V+1))
        has_edge[graph.edge_index[0].cpu(),graph.edge_index[1].cpu()]=1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        null_batch = torch.zeros(1).long().to(device)        
        self.BaseClassifier.eval()
        for data in loader:
            output = self.BaseClassifier(data.x.to(device),data.edge_index.to(device),batch =data.batch.to(device)).cpu().detach()
        Y_labels = np.argmax(output,axis=1)
                
        for i in range(mixed_graph):
            c_graph = mixed_graph[i]
            if Y_labels[i] !=label:
                continue
            E = (c_graph.edge_index.shape[1])//2
            
            exp = self.BaseExplainer.get_explanation_graph(
                x = c_graph.x.to(device),
                edge_index = c_graph.edge_index.to(device),
                label = torch.tensor(label).to(device),
                forward_kwargs = {'batch': null_batch}
                )
            
            mask = exp.edge_imp.cpu()
            
            
            threshold = np.sort(mask)[-int(2*E*self.tau)]
            
            for i in range(c_graph.edge_index.shape[1]):
                u=c_graph.edge_index[0,i]
                v=c_graph.edge_index[1,i]
                if u<v:
                    if mask[i]>=threshold:
                        N[u,v]+=1
                        
        
        edges = graph.edge_index.transpose(1,0)
        
        E_=[]
        
        E_mask = []
        
        E_c = []
        E_c_mask = []
        to_E_c = []
        
        for i in range(V-1):
            for j in range(i+1,V):
                u=nodes[i]
                v=nodes[j]
                E_c.append([u,v])
                E_c_mask.append(N[u,v])
                if has_edge[u,v]==1:
                    E_mask.append(N[u,v])
                    E_.append([u,v])
                    to_E_c.append(len(E_c)-1)
                    
                    
        Es = []
        rank = np.argsort(-np.array(E_mask))
        for i in range(self.k):
            Es.append(E_[rank[i]])
            Es.append([E_[rank[i]][1],E_[rank[i]][0]])
            E_c_mask[to_E_c[rank[i]]] =-1
        rank_other = np.argsort(-np.array(E_c_mask))
        Me = np.zeros(self.k)
        
        for i in range(self.k):
            l = E_[rank[self.k-i-1]]
            h = E_c[rank_other[i]]
            Me[i] = E_mask[rank[self.k-i-1]]-E_c_mask[rank_other[i]]
            if l[0]>h[0]:
                Me[i]-=1
            elif l[0]==h[0] and l[1]>h[1]:
                Me[i]-=1
            Me[i] = Me[i]//2
        return Es, Me


