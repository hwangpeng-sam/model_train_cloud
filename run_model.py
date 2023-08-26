import argparse
import time
import os.path as osp
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score

from utils import linear_split, stationwise_split, EvcFeatureGenerator
from dataset import EvcDataset
from basemodels_multiclass import MultiSeqBase, MultiSeqNormal, MultiSeqUmap, MultiSeqUmapEmb, MultiSeqUmapEmbGating

import pickle
import os


def train(model, train_dataloader, optim, epoch, verbose=0):
    model.train()
    for b_i, (R, H, T, S, y) in enumerate(train_dataloader):
        optim.zero_grad()
        pred = model(R, H, T, S)
        loss = F.nll_loss(pred, y.flatten())
        loss.backward()
        optim.step()
        
        if verbose:
            if b_i % 5000 == 0:
                print('epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}'.format(
                    epoch, b_i * len(R), len(train_dataloader.dataset),
                    100 * b_i / len(train_dataloader), loss.item()
                ))

def test(model, test_dataloader):
    model.eval()

    loss = 0
    with torch.no_grad():
        pred_total = torch.Tensor()
        y_total = torch.Tensor()

        for R, H, T, S, y in test_dataloader:
            pred = model(R, H, T, S)
            loss += F.nll_loss(pred, y.flatten(), reduction='sum').item()
            pred_total = torch.cat((pred_total, pred), dim=0)
            y_total = torch.cat((y_total, y), dim=0)

    # metrics
    loss /= len(test_dataloader.dataset)
    pred_labels = pred_total.argmax(dim=1, keepdim=True)
    pred_exp = torch.exp(pred_total).detach().numpy()

    f1_macro = f1_score(y_total, pred_labels, average='macro')
    f1_micro = f1_score(y_total, pred_labels, average='micro')
    f1_weighted = f1_score(y_total, pred_labels, average='weighted')

    auc_macro = roc_auc_score(y_total.flatten(), pred_exp, average='macro', multi_class='ovr')
    auc_weighted = roc_auc_score(y_total.flatten(), pred_exp, average='weighted', multi_class='ovr')

    recall = recall_score(y_total, pred_labels, labels=[0,1,2], average=None)
    accuracy = accuracy_score(y_total, pred_labels)
    bal_accuracy = balanced_accuracy_score(y_total, pred_labels)

    print('Test dataset:  Loss: {:.4f}, F1: [macro: {:.4f}, micro: {:.4f}, weighted: {:.4f}], Accuracy: {:.4f}, Recalls: [0: {:.2f}, 1: {:.2f}, 2: {:.2f}], Balanced-Accuracy: {:.4f}, AUC: [macro: {:.4f}, weighted: {:.4f}]' \
    .format(loss, f1_macro, f1_micro, f1_weighted, accuracy, recall[0], recall[1], recall[2], bal_accuracy, auc_macro, auc_weighted))

    metrics = {'loss':loss, 'f1_macro':f1_macro, 'f1_micro':f1_micro, 'f1_weighted':f1_weighted, 
               'auc_macro':auc_macro, 'auc_weighted':auc_weighted, 'bal_accuracy':bal_accuracy, 'accuracy':accuracy,
               'recall_0':recall[0], 'recall_1':recall[1], 'recall_2':recall[2]}

    return metrics


def run(args):
    print(f'prediction step is {args.pred_step} ({args.pred_step*20} min) interval')
    np.random.seed(42)  # fix random seed
    input_dir = './model_input_data'
    #Load station attribute, embeddings
    train_seq = pd.read_csv(input_dir + '/train_data_label_630.csv', parse_dates=['time'])
    station_attributes = pd.read_csv(input_dir + '/station_attrs_630.csv')
    station_embeddings = pd.read_csv(input_dir + '/station_embed_630.csv')
    train_generator = EvcFeatureGenerator(train_seq, station_attributes.copy(), station_embeddings.copy())

    #Load R_seq, H_seq, T, S, Y
    data_dir = f'./model_input_data/{args.pred_step}'

    pickle_file_path = os.path.join(data_dir, f'R_seq_{args.pred_step}.pkl')
    with open(pickle_file_path, 'rb') as f:
        R_seq = pickle.load(f)
    pickle_file_path = os.path.join(data_dir, f'H_seq_{args.pred_step}.pkl')
    with open(pickle_file_path, 'rb') as f:
        H_seq = pickle.load(f)

    pickle_file_path = os.path.join(data_dir, f'T_{args.pred_step}.pkl')
    with open(pickle_file_path, 'rb') as f:
        T = pickle.load(f)

    pickle_file_path = os.path.join(data_dir, f'S_{args.pred_step}.pkl')
    with open(pickle_file_path, 'rb') as f:
        S = pickle.load(f)

    pickle_file_path = os.path.join(data_dir, f'Y_{args.pred_step}.pkl')
    with open(pickle_file_path, 'rb') as f:
        Y = pickle.load(f)


    # 4) Train : Valid Split
    train_arrays, valid_arrays = linear_split(R_seq, H_seq, T, S, Y, n_station=630, train_frac=(1-float(args.test_frac)))
    trainset = EvcDataset(*train_arrays)
    testset = EvcDataset(*valid_arrays)
    print(f'Trainset Size: {len(trainset)}, Testset Size: {len(testset)}')


    # 5)  Data Loader
    # with negative over sampling
    labels = trainset[:][-1].flatten().numpy()
    label_counter = Counter(labels)
    weight_encoder = {l:(1_000/cnt) for l,cnt in label_counter.items()}
    weights = list(map(weight_encoder.get, labels))
    num_samples = len(trainset)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)
    train_loader = DataLoader(trainset, batch_size=64, sampler=sampler)
    test_loader = DataLoader(testset, batch_size=1024, shuffle=False)

    models = {'MultiSeqBase':MultiSeqBase, 'MultiSeqNormal':MultiSeqNormal, 'MultiSeqUmap':MultiSeqUmap, 
              'MultiSeqUmapEmb':MultiSeqUmapEmb, 'MultiSeqUmapEmbGating':MultiSeqUmapEmbGating}

    EMB_MODELS = ['MultiSeqUmap', 'MultiSeqUmapEmb', 'MultiSeqUmapEmbGating']
    basemodel = models[args.model]
    print(basemodel)

    result_metrics = []

    print(f'-------{args.model}-------')
    if args.model in EMB_MODELS:
        model = basemodel(n_labels=3, hidden_size=32, embedding_dim=8, pretrained_embedding=train_generator.umap_embedding_vectors, dropout_p=args.dropout_p)
    else:
        model = basemodel(n_labels=3, hidden_size=32, embedding_dim=8, dropout_p=args.dropout_p)
    # optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    # #위에는 timewise_run에 있던 옵티마이저
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optim = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # #아래 옵티마이저는 run_basemodel.py에 있었던 것
    # # optim = torch.optim.Adam(model.parameters(), weight_decay=1e-3)

    for epoch in range(1, args.n_epoch + 1):
        print(f'<<Epoch {epoch}>>', end='\n')
        train(model, train_loader, optim, epoch, verbose=1)
        model_metrics = test(model, test_loader)
        # save after last epoch
        model_metrics.update({'model':args.model, 'test_frac':args.test_frac, 'epoch':epoch})
        result_metrics.append(model_metrics)
        if epoch == 1: #% 20 == 0: #20에폭 단위로 저장
            # Create the directory if it doesn't exist
            directory = f'./model_output/{args.pred_step}'
            print(f'saving file at {directory}')
            if not os.path.exists(directory):
                os.makedirs(directory)

            #모델 저장
            #재학습 안할꺼라 옵티마이저는 저장 안함
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model, 
                           f'{directory}/{args.model}_epoch-{epoch}_pred_step-{args.pred_step}_model.pt')
            print(f'epoch-{epoch}_model saved')
            #결과 저장
            pickle_file_path = os.path.join(directory, f'{args.model}_epoch-{epoch}_pred_step-{args.pred_step}_resultmetric.pkl')
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(result_metrics, f)         
            print(f'epoch-{epoch}_result metric saved')
            
            #임베딩값 일단 무시
            # if args.model in EMB_MODELS:
            #     emb_df = pd.DataFrame(index=train_generator.station_embeddings.sid, data=model.sid_embedding.weight.clone().detach().numpy(), columns=['dim_' + str(i) for i in range(9)])# 수정(8)])
            #     emb_df.to_csv(f'./model_output/embdf_model-{args.model}_testfrac-{args.test_frac}_predstep-{args.pred_step}_epoch-{epoch}.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for running model comparison')
    
    parser.add_argument('--test_frac', default='0.2', type=str)
    parser.add_argument('--n_in', default=12, type=int)
    parser.add_argument('--n_out', default=6, type=int)
    parser.add_argument('--n_hist', default=4, type=int)
    parser.add_argument('--history_smoothing', action='store_true')
    parser.add_argument('--pred_step', default=1, type=int)
    parser.add_argument('--dropout_p', default=0.2, type=float)
    parser.add_argument('--n_epoch', default=1, type=int)
    parser.add_argument('--model', default='MultiSeqUmapEmb', type=str)
    args = parser.parse_args()

    run(args)
