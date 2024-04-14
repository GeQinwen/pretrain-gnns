import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error


criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()



def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred.sigmoid())  # Use sigmoid to convert logits to probabilities

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    metrics = {
        'roc_auc': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'mse': [],
        'mae': []
    }

    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i]**2 > 0
            y_true_valid = (y_true[is_valid, i] + 1) / 2
            y_scores_valid = y_scores[is_valid, i]
            
            metrics['roc_auc'].append(roc_auc_score(y_true_valid, y_scores_valid))
            metrics['precision'].append(precision_score(y_true_valid, y_scores_valid > 0.5))
            metrics['recall'].append(recall_score(y_true_valid, y_scores_valid > 0.5))
            metrics['f1'].append(f1_score(y_true_valid, y_scores_valid > 0.5))
            metrics['mse'].append(mean_squared_error(y_true_valid, y_scores_valid))
            metrics['mae'].append(mean_absolute_error(y_true_valid, y_scores_valid))

    # Compute the average for each metric
    metrics_avg = {metric: np.mean(values) for metric, values in metrics.items()}

    if len(metrics['roc_auc']) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(metrics['roc_auc'])) / y_true.shape[1]))

    return metrics_avg




def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []


    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)
        
    pretrained_metrics = eval(args, model, device, test_loader)
    print(f"Pretrained Test Metrics: AUC: {pretrained_metrics['roc_auc']}, Precision: {pretrained_metrics['precision']}, Recall: {pretrained_metrics['recall']}, F1: {pretrained_metrics['f1']}, MSE: {pretrained_metrics['mse']}, MAE: {pretrained_metrics['mae']}")

    best_test_metrics = pretrained_metrics
    
    for epoch in range(1, args.epochs + 1):
        print(f"====epoch {epoch}")
        
        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_metrics = eval(args, model, device, train_loader)
            print(f"Train Metrics: AUC: {train_metrics['roc_auc']}, Precision: {train_metrics['precision']}, Recall: {train_metrics['recall']}, F1: {train_metrics['f1']}, MSE: {train_metrics['mse']}, MAE: {train_metrics['mae']}")
        else:
            print("omit the training accuracy computation")
            train_metrics = {'roc_auc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'mse': 0, 'mae': 0}
        
        val_metrics = eval(args, model, device, val_loader)
        test_metrics = eval(args, model, device, test_loader)
        print(f"Current epoch test AUC: {test_metrics['roc_auc']} | Diff with baseline: {test_metrics['roc_auc'] - pretrained_metrics['roc_auc']}")
        

        if test_metrics['roc_auc'] > best_test_metrics['roc_auc']:
            best_test_metrics = test_metrics
            
        print(f"Val Metrics: AUC: {val_metrics['roc_auc']}, Precision: {val_metrics['precision']}, Recall: {val_metrics['recall']}, F1: {val_metrics['f1']}, MSE: {val_metrics['mse']}, MAE: {val_metrics['mae']}")
        print(f"Test Metrics: AUC: {test_metrics['roc_auc']}, Precision: {test_metrics['precision']}, Recall: {test_metrics['recall']}, F1: {test_metrics['f1']}, MSE: {test_metrics['mse']}, MAE: {test_metrics['mae']}")

        val_acc_list.append(val_metrics['roc_auc'])
        test_acc_list.append(test_metrics['roc_auc'])
        train_acc_list.append(train_metrics['roc_auc'])

        if args.filename:
            writer.add_scalar('data/train auc', train_metrics['roc_auc'], epoch)
            writer.add_scalar('data/val auc', val_metrics['roc_auc'], epoch)
            writer.add_scalar('data/test auc', test_metrics['roc_auc'], epoch)

            # Add additional metrics to tensorboard
            for metric in ['precision', 'recall', 'f1', 'mse', 'mae']:
                writer.add_scalar(f'data/train {metric}', train_metrics[metric], epoch)
                writer.add_scalar(f'data/val {metric}', val_metrics[metric], epoch)
                writer.add_scalar(f'data/test {metric}', test_metrics[metric], epoch)

        print("")
    print(f"Best Test Set AUC: {best_test_metrics['roc_auc']} | Diff with pretrained: {best_test_metrics['roc_auc'] - pretrained_metrics['roc_auc']}")


    if not args.filename == "":
        writer.close()

if __name__ == "__main__":
    main()
