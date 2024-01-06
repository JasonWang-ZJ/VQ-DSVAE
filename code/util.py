from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
import math
import sys
import torch
import numpy as np
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch.nn as nn
from torch_geometric.utils import coalesce
import scipy.sparse as sp
import pickle as pkl
import networkx as nx


def merge_edges_over_time(graph_data):
    '''
    merge the input edge sets
    '''
    edge_l = [dt.edge_index for dt in graph_data]
    edge_all = torch.concat(edge_l, -1)
    return coalesce(edge_all)


def normalize(val, v_min, v_max):
    '''
    normalize to 0-1
    '''
    return (val-v_min)/(v_max - v_min)


def scale_back(val, v_min, v_max):
    '''
    0-1 back to orignal scale
    '''
    return val*(v_max - v_min) + v_min
    
    
def node_feature_to_adj_list(node_features,max_gap,attend_lane_dist, self_loop = True,  return_weight=False, xy_scale = 1):
    '''
    calculate adjacency list from node features (mileage and lane)
    based on input. Mileage and lane might be normalized
    e.g, for normalized 4 lane, neighboring lane delta ~= 0.33
    node_features -  ["Lane", "Class","x","Speed", "Acceleration","Mileage" ].
    max_gap - max longitudinal distance in miles to attend to.
    attend_lane_dist - max lane distance to attend to.
    return_weight - return edge weight as function of distance
    xy_scale - scaling factor between x, y positions e.g. because of units and normalization.
                             
    '''
    n_nodes = node_features.shape[0]
    lane_all = node_features[:, 2]
    dist_all = node_features[:, -1]
    speed_all = node_features[:, 3]
    lane_adj = np.expand_dims(lane_all,1) - np.expand_dims(lane_all,0)
    dist_adj = np.expand_dims(dist_all,1) - np.expand_dims(dist_all,0)
    # dist_adj[i,j] = dist_i - dist_j, positive means i in front of j
    dist_adj_bool = abs(dist_adj) <= max_gap
    lane_adj_bool = abs(lane_adj) <= attend_lane_dist 
    adj_matrix = lane_adj_bool & dist_adj_bool
    if not self_loop:
        adj_matrix &= ~np.eye(n_nodes, dtype=bool)
    adj_list = [[i,j] for i in range(n_nodes) for j in range(n_nodes) if adj_matrix[i][j] ]
    adj_list = np.array(adj_list)
    edge_dict = {}
    
    if return_weight:
        d_norm = np.sqrt((dist_adj*xy_scale) **2 + lane_adj**2) # l2 norm
        adj_exp = np.exp(-d_norm)
        adj_sim = 1/(d_norm + np.eye(d_norm.shape[0])) # eye to avoid devide by 0

        attr_exp = adj_exp[adj_matrix]
        attr_sim = adj_sim[adj_matrix]
        edge_dict['weight'] = (attr_exp, attr_sim)
    
    return adj_list.T, edge_dict
    

###### loss #####


criterion_CEL = nn.CrossEntropyLoss()
criterion_NLL = nn.GaussianNLLLoss()
criterion_MSE = nn.MSELoss()


def cal_loss(model_output, loss_vq, graph_x, rate_lane, rate_speed=0, rate_acc = 0, rate_vq=0, mask = [],  n_lane = 4):
    '''
    calculate loss for each sample graph, used in training
    model_output, graph_x: [n_node, time, feature], where model_output feature: [mu_dist, sig_dist, p_lane1,..,p_lanen, mu_v, sig_v, mu_a, sig_a]
    rate_lane - loss rate for lane 
    output - a single loss value 
    '''
    # pred
    dist_pred = model_output[:,:,:2]
    lane_pred = model_output[:,:, 2:2+n_lane]
    # true
    lane = graph_x[:,:,[0]].long()
    dist = graph_x[:,:,[-1]]
    # reshape lane
    lane = lane.view(-1) # [node* time]
    lane_pred = lane_pred.view(-1, lane_pred.shape[-1])
    # dist gaussian loss   
    mean_dist = dist_pred[:,:, [0]]
    var_dist = torch.exp(dist_pred[:,:,[1]])

    if len(mask) > 0: 
        # use mask
        loss_nll = criterion_NLL(mean_dist[mask] , dist[mask] , var_dist[mask])
        loss_cel = criterion_CEL(lane_pred[mask.flatten()], lane[mask.flatten()])
    else:
        loss_nll = criterion_NLL(mean_dist, dist, var_dist)
        loss_cel = criterion_CEL(lane_pred, lane)
    loss = loss_nll + rate_lane*loss_cel
    if rate_speed > 0:
        speed_pred = model_output[:,:, 2+n_lane:4+n_lane]
        speed = graph_x[:,:,[3]]
        mean_sp = speed_pred[:,:, [0]]
        var_sp = torch.exp(speed_pred[:,:,[1]])
        if len(mask) > 0:
            loss_speed = criterion_NLL(mean_sp[mask], speed[mask], var_sp[mask])
        else: 
            loss_speed = criterion_NLL(mean_sp, speed, var_sp)
        loss += rate_speed*loss_speed
    if rate_acc > 0:
        acc = graph_x[:,:,[4]]
        acc_pred = model_output[:,:, 4+n_lane:6+n_lane]
        mean_acc = acc_pred[:,:, [0]]
        var_acc = torch.exp(acc_pred[:,:,[1]])
        if len(mask) > 0:
            loss_acc = criterion_NLL(mean_acc[mask], acc[mask], var_acc[mask])
        else:
            loss_acc = criterion_NLL(mean_acc, acc, var_acc)
        loss += rate_acc*loss_acc
    loss += rate_vq * loss_vq
    return loss


NLL_loss = nn.GaussianNLLLoss(reduction = 'none')
CE_loss = nn.CrossEntropyLoss(reduction = 'none')

def cal_loss_car(model_output, graph_x, rate_lane, rate_speed=0, rate_acc = 0,time_agg_loss = True, mask = [], n_lane = 4):
    '''
    calculate loss for each sample graph, used in evaluation
    model_output, graph_x: [n_node, time, feature], where model_output feature: [mu_dist, sig_dist, p_lane1,..,p_lanen, mu_v, sig_v, mu_a, sig_a]
    output - [n_car, time] if time_agg_loss = False; else [n_car,]
    '''
    # pred
    dist_pred = model_output[:,:,:2]
    lane_pred = model_output[:,:, 2:2+n_lane]
    # true
    lane = graph_x[:,:,[0]].long()
    dist = graph_x[:,:,[-1]]
    # reshape lane
    lane = lane.view(-1) # [node* time]
    lane_pred = lane_pred.view(-1, lane_pred.shape[-1])
    # dist gaussian loss   
    mean_dist = dist_pred[:,:, [0]]
    var_dist = torch.exp(dist_pred[:,:,[1]])

    n_car = len(graph_x)
    loss_nll = NLL_loss(mean_dist, dist, var_dist).squeeze(-1) # [n_car, time]
    loss_cel = CE_loss(lane_pred, lane).view(n_car,-1) # [n_car, time]  
    loss = loss_nll + rate_lane * loss_cel

    # speed
    if rate_speed > 0:
        speed = graph_x[:,:,[3]]
        speed_pred = model_output[:,:, 2+n_lane:4+n_lane]
        mean_s = speed_pred[:,:, [0]]
        var_s = torch.exp(speed_pred[:,:,[1]])
        loss_speed = NLL_loss(mean_s, speed, var_s).squeeze(-1) # [n_car, time]
        loss += rate_speed*loss_speed

    # acc
    if rate_acc > 0:
        acc = graph_x[:,:,[4]]
        acc_pred = model_output[:,:, 4+n_lane:6+n_lane]
        mean_acc = acc_pred[:,:, [0]]
        var_acc = torch.exp(acc_pred[:,:,[1]])
        loss_acc = NLL_loss(mean_acc, acc, var_acc).squeeze(-1) # [n_car, time]
        loss += rate_acc*loss_acc
        
    if len(mask) > 0:
        loss[~mask] = np.nan

    if time_agg_loss:
        # loss per car across all time steps
        loss = loss.nanmean(1)
    return loss

    
    
###### functions for bivariate normal trajectory modeling ######
    
def pred_to_distribution(traj_pred):
    '''
    from model_output distribution to torch multivariate Normal distribution
    traj_pred - [n_node, time_step, dim_feat2], where dim_feat2 = 5 -> [mu_x, mu_y, std_x, std_y, corr]
    '''
#     traj_pred = traj_pred.permute(0,2,1) #[node, feat,time] to [node, time, feat]
    sx = torch.exp(traj_pred[:,:,2]) #sx
    sy = torch.exp(traj_pred[:,:,3]) #sy
    corr = torch.tanh(traj_pred[:,:,4]) #corr

    cov = torch.zeros(traj_pred.shape[0],traj_pred.shape[1],2,2).to(traj_pred.device)
    cov[:,:,0,0]= sx*sx
    cov[:,:,0,1]= corr*sx*sy
    cov[:,:,1,0]= corr*sx*sy
    cov[:,:,1,1]= sy*sy
    mean = traj_pred[:,:,0:2]
    
    return MultivariateNormal(mean,cov)


def bivariate_loss(traj, traj_pred):
    '''
    traj - [n_node, time_step, dim_feat1], where dim_feat = 2 -> [x,y]
    traj_pred - [n_node, time_step, dim_feat2], where dim_feat2 = 5 -> [mu_x, mu_y, std_x, std_y, corr]
    '''
    dist = pred_to_distribution(traj_pred)
    loss = -dist.log_prob(traj).mean().mean()
    return loss


###### functions for MMA scalars ######

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']

    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    # print(test_idx_range)
    # print("test_idx_reorder", len(test_idx_reorder))

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = list(range(min(test_idx_reorder), max(test_idx_reorder) + 1))
        # print(set(test_idx_range_full)-set(test_idx_range))
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    # print(allx.shape, tx.shape)
    # print(features.shape)

    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    # print("labels", labels.shape)
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # print("labels", labels.shape)

    idx_test = test_idx_range.tolist()

    if dataset == 'cora':

        idx_train = range(len(y) + 1068)
        idx_val = range(len(y) + 1068, len(y) + 1068 + 500)


    elif dataset == 'citeseer':

        idx_train = range(len(y) + 1707)
        idx_val = range(len(y) + 1707, len(y) + 1707 + 500)


    elif dataset == 'pubmed':
        idx_train = range(len(y) + 18157)
        idx_val = range(len(y) + 18157, len(y) + 18157 + 500)

    ## find each node's neighbors
    add_all = []
    for i in range(adj.shape[0]):
        add_all.append(adj[i].nonzero()[1])

    features = torch.FloatTensor(np.array(features.todense()))
    # print(labels)
    if dataset == "citeseer":
        new_labels = []
        for lbl in labels:
            lbl = np.where(lbl == 1)[0]
            new_labels.append(lbl[0] if list(lbl) != [] else 0)
        labels = torch.LongTensor(new_labels)
    else:
        labels = torch.LongTensor(np.where(labels)[1])

    # print("labels", labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return add_all, adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def edge_indices_to_adj(edge_indices):
    # 为了创建一个邻接矩阵，我们需要为每条边分配一个值，这里我们使用1
    values = torch.ones(edge_indices.shape[1])

    # 我们需要知道节点的总数来确定邻接矩阵的大小
    num_nodes = torch.max(edge_indices) + 1

    # 创建稀疏邻接矩阵
    adjacency_matrix = torch.sparse_coo_tensor(edge_indices, values, (num_nodes, num_nodes))

    return adjacency_matrix
