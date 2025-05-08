import torch
import numpy as np
import matplotlib.pyplot as plt
#from graphxai.datasets import ShapeGGen
#from graphxai.gnn_models.node_classification import train, test
#from graphxai.gnn_models.node_classification.testing import GCN_3layer,GIN_3layer,GSAGE_3layer,JKNet_3layer
# Get dataset:
    
import os
from tqdm import tqdm

from graphxai.datasets import FluorideCarbonyl as FC
from graphxai.datasets import Benzene
from graphxai.datasets import MUTAG,DD
from graphxai.datasets import AlkaneCarbonyl as AC
from graphxai.datasets import BA3Motif

from graphxai.datasets import BAHouse,BADiamond,BAWheel,BACycle

from torch_geometric.data import Data
from graphxai.explainers import GuidedBP,PGExplainer,PGCExplainer,SubExplainer, IntegratedGradExplainer,GNNExplainer,SubgraphX
from graphxai.metrics import graph_exp_acc,graph_exp_edge_acc
from graphxai.explainers import ReFine
from graphxai.explainers import GSAT


from graphxai.gnn_models.graph_classification import train, test,test_multi
from graphxai.gnn_models.graph_classification.gcn import GCN_2layer, GCN_3layer
from graphxai.gnn_models.graph_classification.gin import GIN_2layer, GIN_3layer

from robust_p import RobustClassifier,HashAgent,enlarge_graph,RobustExplainer

device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 1200
import statistics
import random
T=70
p=0.3
P='03'
def permutation(dataset):
    for i in range(len(dataset.graphs)):
        graph = dataset.graphs[i]
        nodes= graph.x.shape[0]
        index =[i for i in range(nodes)]
        random.shuffle(index)
        permutation = np.array(index)
        graph.x = graph.x[permutation]
        graph.edge_index[0]=torch.tensor(permutation[graph.edge_index[0]])
        graph.edge_index[1]=torch.tensor(permutation[graph.edge_index[1]])
        dataset.graphs[i] = graph
        
Hasher = HashAgent(h = "md5",T=T,p=p)
print("OK")
# Load data: ------------------------------------------
#dataset = FC(split_sizes = (0.7, 0.2, 0.1), seed = seed)
#dataset = BAHouse(split_sizes = (0.7, 0.2, 0.1), seed = seed)
#dataset = BADiamond(split_sizes = (0.7, 0.2, 0.1), seed = seed)
#dataset = BAWheel(split_sizes = (0.7, 0.2, 0.1), seed = seed)
dataset = Benzene(split_sizes = (0.7, 0.2, 0.1), seed = seed)
#dataset = FC(split_sizes = (0.7, 0.2, 0.1), seed = seed)

#model_path = "./checkpoint/classifier/GCN-Benzene-T={}--md5.pth"
#model_path = "./checkpoint/classifier/GCN3-Benzene-original.pth"
model_path = "./checkpoint/classifier-p/GCN-Benzene-T={}-p={}.pth".format(T,P)
#model_path = "./checkpoint/classifier-p/GCN-DD-T={}-p={}.pth".format(T,P)

print(device)
if os.path.exists(model_path):            
    model = torch.load(model_path, weights_only=False)
else:
    #model = GCN_3layer(10, 128, 2).to(device)
    #model = GCN_3layer(89, 128, 2).to(device)
    model = GCN_3layer(14, 128, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    maxf1 =0
    en_dataset = enlarge_graph(dataset,Hasher)
    #en_dataset = dataset
    train_loader, _ =  en_dataset.get_train_loader(batch_size = 2048)
    val_loader, _ = en_dataset.get_val_loader()
    test_loader, _ = en_dataset.get_test_loader()
    for epoch in tqdm(range(1, 1000)):
        train(model, optimizer, criterion, train_loader,device = device)
        f1, prec, rec, auprc, auroc = test(model, val_loader,device =device)
        #f1, prec, rec, auprc, auroc = test_multi(model, test_loader,device =device)
        print(f'Epoch: {epoch:03d}, Val F1: {f1:.4f}, Val rec: {rec:.4f}')
        if f1>maxf1:
            torch.save(model, model_path)
            maxf1=f1     
    model = torch.load(model_path)



#train Explainer on the original dataset



robust_classifier = RobustClassifier(model,Hasher)


dataset = Benzene (split_sizes = (0.7, 0.2, 0.1), seed = seed)
#dataset = BADiamond(split_sizes = (0.7, 0.2, 0.1), seed = seed)



#null_batch = [1]
null_batch = torch.zeros(1).long().to(device)
forward_kwargs = {'batch': null_batch} # Input to explainers forward methods

#Classifier test
correct=0
test_datas, gt_exps = dataset.get_test_list()
wrong = 0
for t in range(len(test_datas)):
    print(f"{t}/{len(test_datas)}")
    test_data = test_datas[t]
    #gt_exp = gt_exps[t]
    
    gt_exp = gt_exps[t]
    #gt_exp = [gt_exps[t]]
    #test_data, gt_exp = dataset.get_test_w_label(label = 1) # Get positive example
    test_data = test_data.to(device)
# Predict on the chosen testing sample: ----------------
# Filler to get around batch variable in GNN model
    
    
    model.eval()    
     
    vote_label, votes = robust_classifier.vote(test_data)
    with torch.no_grad():
        prediction = model(test_data.x, test_data.edge_index, batch = null_batch)
    print(votes)    
    predicted = prediction.argmax(dim=1).item()
    #if 1-test_data.y==1 and len(votes)==1:
    #    wrong+=0
    #    print(0)
    #else:
    #    wrong+=votes[1-test_data.y]/np.sum(votes)
    #    print(votes[1-test_data.y]/np.sum(votes))
    print(test_data.y, vote_label, predicted)
    if vote_label!=test_data.y:
        continue
    else:
        correct+=1
#print(wrong/len(test_datas))
print(correct/len(test_datas))




test_loader, _ = dataset.get_test_loader()

"""
refine = ReFine(model, emb_layer_name = 'conv3', coeff_size=0.0001,gamma=0.15,#eps=1e-8,#coeff_ent=1e-10,
    max_epochs=5, lr=0.01, explain_graph = True,in_channels=14)

refine_path = "./checkpoint/explainer-p/RF-GCN3-Benzene-T={}-p={}.pth".format(T,P)

#if os.path.exists(pg_explainer_path):            
#    pg_explainer.elayers = torch.load(pg_explainer_path)
#else:
    #train_data, train_exp = dataset.get_train_list()
train_data, train_exp = dataset.get_train_w_labels(label=1)
    #train_data, train_exp = dataset.get_all_w_labels(label=1)
refine.train_explanation_model(train_data, forward_kwargs = forward_kwargs)
torch.save(refine.elayers, refine_path)
"""
#"""
pg_explainer = PGExplainer(model, emb_layer_name = 'conv3', coeff_size=1e-5,coeff_ent=3e-4, eps=1e-16,
    max_epochs=30, lr=0.01, explain_graph = True)


pg_explainer_path = "./checkpoint/explainer/Pg-GCN-benzene-T={}-p={}.pth".format(T,P)
#pg_explainer_path = "./checkpoint/explainer/Pg-GIN3-BA3Motif-T=50-p=05.pth"

if os.path.exists(pg_explainer_path):            
    pg_explainer.elayers = torch.load(pg_explainer_path)
else:
    #pg_explainer.train_explanation_model(dataset.graphs, forward_kwargs = forward_kwargs)
    #torch.save(pg_explainer.elayers, pg_explainer_path)
    #en_dataset = enlarge_graph(dataset,Hasher,split_sizes = (0.8, 0.2, 0))
    train_data, train_exp = dataset.get_train_list()
    pg_explainer.train_explanation_model(train_data, forward_kwargs = forward_kwargs)
        #sub_explainer.train_explanation_model(dataset.graphs, forward_kwargs = forward_kwargs)
        #sub_explainer.train_explanation_model(en_dataset.graphs, forward_kwargs = forward_kwargs)
    torch.save(pg_explainer.elayers, pg_explainer_path)
#"""
"""
gsat = GSAT(model, emb_layer_name = 'conv3', coeff_size=1e-3,coeff_ent=1e-7, eps=1e-16,
    max_epochs=5, lr=0.01, explain_graph = True)
#Diamond:coeff_size=0.000001, 
#0.0003
gsat_path = "./checkpoint/explainer/GSAT-GCN-frac-FC-T={}-p={}.pth".format(T,P)
#pg_explainer_path = "./checkpoint/explainer/Pg-GIN3-BA3Motif-T=50-p=05.pth"

#if os.path.exists(pg_explainer_path):            
#    pg_explainer.elayers = torch.load(pg_explainer_path)
#else:
    #pg_explainer.train_explanation_model(dataset.graphs, forward_kwargs = forward_kwargs)
    #torch.save(pg_explainer.elayers, pg_explainer_path)
    #en_dataset = enlarge_graph(dataset,Hasher,split_sizes = (0.8, 0.2, 0))
train_data, train_exp = dataset.get_train_w_labels(label=0)
gsat.train_explanation_model(train_data, forward_kwargs = forward_kwargs)
        #sub_explainer.train_explanation_model(dataset.graphs, forward_kwargs = forward_kwargs)
        #sub_explainer.train_explanation_model(en_dataset.graphs, forward_kwargs = forward_kwargs)
torch.save(gsat.elayers, gsat_path)

#sub_explainer_r = SubExplainer(model, emb_layer_name = 'conv3', 
#    max_epochs=10, lr=0.03, explain_graph = True)

#f1, prec, rec, auprc, auroc = test_multi(model, test_loader,device =device)
#print(f'Test F1: {f1:.4f}, Test AUROC: {auroc:.4f}')
"""


#GBP = GuidedBP(model)




"""
test_datas, gt_exps = dataset.get_all_w_labels(label = 1)
print(len(test_datas))
"""


testtimes = 2000
cnt=0

pg_results = []
rpg_results = []


pg_accs = []
rpg_accs = []

Ms_p = []

GBP_results = []
rGBP_results = []


GBP_accs = []
rGBP_accs = []

Ms_gbp = []



gsat_results = []
rgsat_results = []

gsat_accs = []
rgsat_accs = []

Ms_g = []

rf_results = []
rrf_results = []
Ms_r = []
rf_accs = []
rrf_accs = []

#Hasher2 = HashAgent(h = "md5",T=T,p=0.5)
#rpg_explainer = RobustExplainer(pg_explainer,model,Hasher2)
#r_refine = RobustExplainer(refine,model,Hasher)
#r_refine.set_para(tau=0.3)
rpg_explainer = RobustExplainer(pg_explainer,model,Hasher)
rpg_explainer.set_para(tau=0.3)
#rpgc_explainer = RobustExplainer(pgc_explainer,model,Hasher)
#rpgc_explainer.set_para(tau=0.1)
#rGBP = RobustExplainer(GBP,model,Hasher)
#rgsat = RobustExplainer(gsat,model,Hasher)
#rGBP.set_para(tau=0.3)
test_datas, gt_exps = dataset.get_test_w_labels(label = 1)
#test_datas, gt_exps = dataset.get_all_w_labels(label = 1)
#for t in range(0):
for t in range(len(test_datas)):
    if cnt>=testtimes:
        break
    print(f"{cnt}/{testtimes}")
    print(f"{t}/{len(test_datas)}")
    test_data = test_datas[t]
    gt_exp = gt_exps[t]
    #gt_exp = [gt_exps[t]]
    if np.sum(np.array(gt_exp[0].edge_imp))<1:
        continue
    
    #test_data, gt_exp = dataset.get_test_w_label(label = 1) # Get positive example
    test_data = test_data.to(device)
# Predict on the chosen testing sample: ----------------
# Filler to get around batch variable in GNN model
    
    k=12
    
    if test_data.edge_index.shape[1]<=k:
        continue
    
    #if len(gt_exp)>1:
    #    continue
    
    model.eval()    
    
    vote_label, Mc = robust_classifier.vote(test_data)
    with torch.no_grad():
        prediction = model(test_data.x, test_data.edge_index, batch = null_batch)

    predicted = prediction.argmax(dim=1).item()
    print(test_data.y, vote_label, predicted)
    if vote_label!=test_data.y:
        continue
    cnt+=1
    
    
    #Obtain the ground truth explanation edges
    graph = test_data.edge_index.transpose(1,0)
    
    #"""
    true_edges= []
    k_real = 0
    for j in range(len(gt_exp)):
        gt_graph = gt_exp[j].graph.edge_index.transpose(1,0)
        relative_positives = (gt_exp[j].edge_imp == 1).nonzero(as_tuple=True)[0].cpu()
        for i in relative_positives:
            flag=0
            for ks in range(len(true_edges)):
                
                if gt_graph[i][0].cpu() == true_edges[ks][0] and gt_graph[i][1].cpu() == true_edges[ks][1]:
                    flag=1
                
            if flag==0:
                true_edges.append(gt_graph[i].cpu())
    k_real = len(true_edges)
    print(k_real)
    #"""
    """
    gt_exp = gt_exp[0]
    gt_graph = gt_exp.graph.edge_index.transpose(1,0)
    #print(gt_exp.edge_imp)
    #print(gt_exp.node_imp)
    #print(test_data.x)
    #print(test_data.x)
    relative_positives = (gt_exp.edge_imp == 1).nonzero(as_tuple=True)[0].cpu()
    k_real = len(relative_positives)
    true_edges = [gt_graph[i].cpu() for i in relative_positives]
    """
    #k=k_real
    pg_result = []
    rpg_result = []

    
    GBP_result = []
    rGBP_result = []

    gsat_result = []
    rgsat_result = []
    
    rf_result = []
    rrf_result = []
    """
    GBP_generated_exp = GBP.get_explanation_graph(
        x = test_data.x,
        edge_index = test_data.edge_index,
        label = test_data.y,
        forward_kwargs = forward_kwargs
        )
    
    GBP_edge_imp = GBP_generated_exp.edge_imp.cpu()
    """
    """
    rf_generated_exp = refine.get_explanation_graph(
        x = test_data.x,
        edge_index = test_data.edge_index,
        label = test_data.y,
        forward_kwargs = forward_kwargs
        )
    
    
    rf_edge_imp = rf_generated_exp.edge_imp.cpu()
    """
    #"""
    pg_generated_exp = pg_explainer.get_explanation_graph(
        x = test_data.x,
        edge_index = test_data.edge_index,
        label = test_data.y,
        forward_kwargs = forward_kwargs
        )
    
    
    pg_edge_imp = pg_generated_exp.edge_imp.cpu()
    #"""
    """
    pgc_generated_exp = pgc_explainer.get_explanation_graph(
        x = test_data.x,
        edge_index = test_data.edge_index,
        label = test_data.y,
        forward_kwargs = forward_kwargs
        )
    
    
    pgc_edge_imp = pgc_generated_exp.edge_imp.cpu()
    
    """
    """
    gsat_generated_exp = gsat.get_explanation_graph(
        x = test_data.x,
        edge_index = test_data.edge_index,
        label = test_data.y,
        forward_kwargs = forward_kwargs
        )
    
    gsat_edge_imp = gsat_generated_exp.edge_imp.cpu()
    """
    
    #positive = np.argsort(np.array(-rf_edge_imp))[0:k]
    #positive = np.argsort(np.array(-pgc_edge_imp))[0:k]
    positive = np.argsort(np.array(-pg_edge_imp))[0:k]
    #positive = np.argsort(np.array(-gsat_edge_imp))[0:k]
    #positive = np.argsort(np.array(-GBP_edge_imp))[0:k]
    calc_edge = [graph[i].cpu() for i in positive]
    
    num =0 
    for j in range(len(calc_edge)):
        edge = calc_edge[j]
        for t in range(len(true_edges)):
            true_edge = true_edges[t]
            if true_edge[0]==edge[0] and true_edge[1]==edge[1]:
                num+=1
                
    #rf_results.append("{}:{}/{}".format(k,num,k_real))
    #pg_accs.append(num/k_real)
    #rf_accs.append(num/k)
    pg_results.append("{}:{}/{}".format(k,num,k_real))
    #pg_accs.append(num/k_real)
    pg_accs.append(num/k)
    
    #pgc_results.append("{}:{}/{}".format(k,num,k_real))
    #pgc_accs.append(num/k_real)
    #pgc_accs.append(num/k)
    
    #gsat_results.append("{}:{}/{}".format(k,num,k_real))
    #sub_accs.append(num/k_real)
    #gsat_accs.append(num/k)
    
    #GBP_results.append("{}:{}/{}".format(k,num,k_real))
    #sub_accs.append(num/k_real)
    #GBP_accs.append(num/k)
    
    
    
    #rrf_exp,Me_r = r_refine.explain_overlap(test_data,vote_label,Mc,k//2)
    rpg_exp,Me_p = rpg_explainer.explain_overlap(test_data,vote_label,Mc,k//2)
    #rpgc_exp,Me_pgc = rpgc_explainer.explain_overlap(test_data,vote_label,Mc,k//2)
    #rgsat_exp,Me_g = rgsat.explain_overlap(test_data,vote_label,Mc,k//2)
    #rpg_exp,Me_p = rpg_explainer.explain_with_gt(test_data,vote_label,true_edges,Mc,k//2)
    #rGBP_exp,Me_bp = rGBP.explain_overlap(test_data,vote_label,Mc,k//2)
    
    
    print(true_edges)
    print(calc_edge)
    """
    print(rrf_exp)
    rrf_num =0 
    for j in range(len(rrf_exp)):
        edge = rrf_exp[j]
        for t in range(len(true_edges)):
            true_edge = true_edges[t]
            if true_edge[0]==edge[0] and true_edge[1]==edge[1]:
                rrf_num+=1
    rrf_results.append("{}:{}/{}".format(k,rrf_num,k_real))
    #rpg_accs.append(rpg_num/k_real)
    rrf_accs.append(rrf_num/k)
    """
    #"""
    print(rpg_exp)
    rpg_num =0 
    for j in range(len(rpg_exp)):
        edge = rpg_exp[j]
        for t in range(len(true_edges)):
            true_edge = true_edges[t]
            if true_edge[0]==edge[0] and true_edge[1]==edge[1]:
                rpg_num+=1
    rpg_results.append("{}:{}/{}".format(k,rpg_num,k_real))
    #rpg_accs.append(rpg_num/k_real)
    rpg_accs.append(rpg_num/k)
    #"""
    """
    print(rpgc_exp)
    rpgc_num =0 
    for j in range(len(rpgc_exp)):
        edge = rpgc_exp[j]
        for t in range(len(true_edges)):
            true_edge = true_edges[t]
            if true_edge[0]==edge[0] and true_edge[1]==edge[1]:
                rpgc_num+=1
    rpgc_results.append("{}:{}/{}".format(k,rpgc_num,k_real))
    #rpg_accs.append(rpg_num/k_real)
    rpgc_accs.append(rpgc_num/k)
    """
    """
    rgsat_num =0 
    for j in range(len(rgsat_exp)):
        edge = rgsat_exp[j]
        for t in range(len(true_edges)):
            true_edge = true_edges[t]
            if true_edge[0]==edge[0] and true_edge[1]==edge[1]:
                rgsat_num+=1
    rgsat_results.append("{}:{}/{}".format(k,rgsat_num,k_real))
    #rsub_accs.append(rsub_num/k_real)
    rgsat_accs.append(rgsat_num/k)
    #rsub_accs.append(rsub_acc)
    """
    """
    rGBP_num =0 
    for j in range(len(rGBP_exp)):
        edge = rGBP_exp[j]
        for t in range(len(true_edges)):
            true_edge = true_edges[t]
            if true_edge[0]==edge[0] and true_edge[1]==edge[1]:
                rGBP_num+=1
    rGBP_results.append("{}:{}/{}".format(k,rGBP_num,k_real))
    #rsub_accs.append(rsub_num/k_real)
    rGBP_accs.append(rGBP_num/k)
    #rsub_accs.append(rsub_acc)
    """
    
    #M = min(Mc,Me)
    #Ms_g.append(Me_g)
    #Ms_r.append(Me_r)
    Ms_p.append(Me_p)
    #Ms_pgc.append(Me_pgc)
    #Ms_gbp.append(Me_bp)

"""
print("GBP:")
print(GBP_results)
print(sum(GBP_accs)/len(GBP_accs))
#print("Robust GNN:")
#print(rgnn_results)
"""
"""
print("REFINE:")
print(rf_results)
print(sum(rf_accs)/len(rf_accs))
"""
#"""    
print("PGEX:")
print(pg_results)
print(sum(pg_accs)/len(pg_accs))
#"""
"""
print("PGCEX:")
print(pgc_results)
print(sum(pgc_accs)/len(pgc_accs))
"""
"""
print("GSAT:")
print(gsat_results)
print(sum(gsat_accs)/len(gsat_accs))
#print("Robust GNN:")
#print(rgnn_results)
"""
"""
print("Robust GBP:")
print(rGBP_results)
print(sum(rGBP_accs)/len(rGBP_accs))
#print("M-gnn:")
#print(np.array(Ms_g).transpose(1,0))
"""
"""
print("Robust RF:")
print(rrf_results)
print(sum(rrf_accs)/len(rrf_accs))
"""
#"""
print("Robust PG:")
print(rpg_results)
print(sum(rpg_accs)/len(rpg_accs))
#"""
"""
print("Robust PGC:")
print(rpgc_results)
print(sum(rpgc_accs)/len(rpgc_accs))
"""
"""
print("Robust GSAT:")
print(rgsat_results)
print(sum(rgsat_accs)/len(rgsat_accs))
#print("M-gnn:")
#print(np.array(Ms_g).transpose(1,0))
"""
"""
print("M-GBP:")
Ms_gbp = np.sort(np.array(Ms_gbp).transpose(1,0),axis=1)
for i in range(Ms_gbp.shape[0]):
    print(Ms_gbp[i])
    print(sum(Ms_gbp[i])/len(Ms_gbp[i]))
"""    
"""
print("M-rf")
Ms_r = np.sort(np.array(Ms_r).transpose(1,0),axis=1)
for i in range(Ms_r.shape[0]):
    print(Ms_r[i])
    print(sum(Ms_r[i])/len(Ms_r[i]))
"""
#"""
print("M-pg:")
Ms_p = np.sort(np.array(Ms_p).transpose(1,0),axis=1)
for i in range(Ms_p.shape[0]):
    print(Ms_p[i])
    print(sum(Ms_p[i])/len(Ms_p[i]))
#"""
"""
print("M-pgc:")
Ms_pgc = np.sort(np.array(Ms_pgc).transpose(1,0),axis=1)
for i in range(Ms_pgc.shape[0]):
    print(Ms_pgc[i])
    print(sum(Ms_pgc[i])/len(Ms_pgc[i]))
"""
"""
print("M-gsat:")
Ms_g = np.sort(np.array(Ms_g).transpose(1,0),axis=1)
for i in range(Ms_g.shape[0]):
    print(Ms_g[i])
    print(sum(Ms_g[i])/len(Ms_g[i]))
"""    
