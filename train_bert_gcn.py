import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
from sklearn.metrics import accuracy_score
import numpy as np
import os
import random
import shutil
import argparse
import sys
import logging
from datetime import datetime
from torch.optim import lr_scheduler
from model import BertGCN, BertGAT,BertLSTMGCN,LSTMGCN
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import sentencepiece as spm
# 设置随机数种子函数
torch.autograd.set_detect_anomaly(True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    th.backends.cudnn.deterministic = True  # 确保每次计算相同
    th.backends.cudnn.benchmark = False     # 禁用CuDNN的自动优化
parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('-m', '--m', type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
parser.add_argument('--nb_epochs', type=int, default=50)
parser.add_argument('--bert_init', type=str, default='bertmodel',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased', 'hfl/chinese-roberta-wwm-ext','bertmodel','zhuang_bert'])
parser.add_argument('--pretrained_bert_ckpt', default=None)
parser.add_argument('--dataset', default='20ng', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr','zhuang'])
parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
parser.add_argument('--gcn_model', type=str, default='lstm_gcn', choices=['gcn', 'gat','lstm_gcn'])
parser.add_argument('--gcn_layers', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=200, help='the dimension of gcn hidden layer, the dimension for gat is n_hidden * heads')
parser.add_argument('--heads', type=int, default=8, help='the number of attentionn heads for gat')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--gcn_lr', type=float, default=1e-3)
parser.add_argument('--bert_lr', type=float, default=2e-5)
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')  
parser.add_argument('--alpha', type=int, default=4, help='loss')
args = parser.parse_args()
set_seed(args.seed)
max_length = args.max_length
batch_size = args.batch_size
m = args.m
nb_epochs = args.nb_epochs
bert_init = args.bert_init
pretrained_bert_ckpt = args.pretrained_bert_ckpt
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir
gcn_model = args.gcn_model
gcn_layers = args.gcn_layers
n_hidden = args.n_hidden
heads = args.heads
dropout = args.dropout
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr
alpha = args.alpha

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

cpu = th.device('cpu')
gpu = th.device('cuda:0')

logger.info('arguments:')
logger.info(str(args))
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))
# Model


# Data Preprocess
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
'''
adj: n*n sparse adjacency matrix
y_train, y_val, y_test: n*c matrices 
train_mask, val_mask, test_mask: n-d bool array
'''

# compute number of real train/val/test/word nodes and number of classes
nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

# instantiate model according to class number
if gcn_model == 'gcn':
    model = BertGCN(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    n_hidden=n_hidden, dropout=dropout)  
elif gcn_model ==  'lstm_gcn':
    model = BertLSTMGCN(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    n_hidden=n_hidden, dropout=dropout)
else:
    model = BertGAT(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    heads=heads, n_hidden=n_hidden, dropout=dropout)


if pretrained_bert_ckpt is not None:
    ckpt = th.load(pretrained_bert_ckpt, map_location=gpu)
    model.bert_model.load_state_dict(ckpt['bert_model'])
    model.classifier.load_state_dict(ckpt['classifier'])


# load documents and compute input encodings
corpse_file = './data/corpus/' + dataset +'_shuffle.txt'
with open(corpse_file, 'r') as f:
    text = f.read()
    text = text.replace('\\', '')
    text = text.split('\n')

def encode_input_LSTM(text_list, spm):
    all_encode_ids = []
    all_input_masks = []
    
    for text in text_list:
        encode_ids = spm.encode_as_ids(text)
        if len(encode_ids) > max_length:
            encode_ids = encode_ids[:max_length]
        else:
            encode_ids += [0] * (max_length - len(encode_ids))  # 补齐

        input_masks = [1 if id != 0 else 0 for id in encode_ids]

        all_encode_ids.append(encode_ids)
        all_input_masks.append(input_masks)

    return th.tensor(all_encode_ids), th.tensor(all_input_masks)
def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
#     print(input.keys())
    return input.input_ids, input.attention_mask

input_ids, attention_mask = encode_input(text, model.tokenizer)
input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
attention_mask = th.cat([attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])

# transform one-hot label to class ID for pytorch computation
y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)

# document mask used for update feature
doc_mask  = train_mask + val_mask + test_mask

# build DGL Graph
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
g.ndata['label_train'] = th.LongTensor(y_train)
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

logger.info('graph information:')
logger.info(str(g))

# create index loader
train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
test_idx = Data.TensorDataset(th.arange(nb_node-nb_test, nb_node, dtype=th.long))
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

# Training
def update_feature():
    global model, g, doc_mask
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=1024
    )
    if gcn_model == 'gcnlstm':
        with th.no_grad():
            model = model.to(gpu)
            model.eval()
            cls_list = []
            for i, batch in enumerate(dataloader):
                input_ids, attention_mask = [x.to(gpu) for x in batch]
                embeddings = model.embedding(input_ids)
                output = model.bilstm(embeddings)[0][:, -1, :]
                cls_list.append(output.cpu())
            cls_feat = th.cat(cls_list, axis=0)
    else:
        with th.no_grad():
            model = model.to(gpu)
            model.eval()
            cls_list = []
            for i, batch in enumerate(dataloader):
                input_ids, attention_mask = [x.to(gpu) for x in batch]
                bert_output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
                output = model.bilstm(bert_output)[0][:, -1, :]
                cls_list.append(output.cpu())
            cls_feat = th.cat(cls_list, axis=0)
    g = g.to(cpu)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g

if gcn_model == 'gcnlstm':
    
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) #AdamW优化器
else:
    optimizer = th.optim.Adam([
        {'params': model.bert_model.parameters(), 'lr': bert_lr},
        {'params': model.classifier.parameters(), 'lr': bert_lr},
        {'params': model.gcn.parameters(), 'lr': gcn_lr},
    ], lr=1e-3
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


print(model)
def train_step(engine, batch):
    global model, g, optimizer
    
    model.train()
    model = model.to(gpu)
    g = g.to(gpu)
    optimizer.zero_grad()
    (idx, ) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    y_pred = model(g, idx)[train_mask]
    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    return train_loss, train_acc


trainer = Engine(train_step)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()


def test_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(g, idx)
        y_true = g.ndata['label'][idx]
        return y_pred, y_true



evaluator = Engine(test_step)
metrics={
    'acc': Accuracy(),
    'nll': Loss(th.nn.NLLLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]

    # 计算 precision, recall, F1-score
    y_pred, y_true = evaluator.state.output
    train_precision = precision_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average='macro', zero_division=0.0)
    train_recall = recall_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average='macro')
    train_f1 = f1_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average='macro')

    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]

    y_pred, y_true = evaluator.state.output
    val_precision = precision_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average='macro')
    val_recall = recall_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average='macro')
    val_f1 = f1_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average='macro')

    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]

    y_pred, y_true = evaluator.state.output
    test_precision = precision_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average='macro')
    test_recall = recall_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average='macro')
    test_f1 = f1_score(y_true.cpu(), y_pred.argmax(dim=1).cpu(), average='macro')

    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f} precision: {:.4f} recall: {:.4f} F1: {:.4f} \n"
        "Val acc: {:.4f} loss: {:.4f} precision: {:.4f} recall: {:.4f} F1: {:.4f} \n"
        "Test acc: {:.4f} loss: {:.4f} precision: {:.4f} recall: {:.4f} F1: {:.4f}"
        .format(
            trainer.state.epoch, train_acc, train_nll, train_precision, train_recall, train_f1,
            val_acc, val_nll, val_precision, val_recall, val_f1,
            test_acc, test_nll, test_precision, test_recall, test_f1
        )
    )

    if test_acc > log_training_results.best_test_acc:
        logger.info("New checkpoint")
        if gcn_model == 'gcnlstm':
            th.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
                    }, os.path.join(ckpt_dir, 'pytorch_model.bin')
                    )
        else:
            th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'gcn': model.gcn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
            
        log_training_results.best_test_acc = test_acc


log_training_results.best_test_acc = 0
g = update_feature()
trainer.run(idx_loader, max_epochs=nb_epochs)
logger.info(f"Best Test Accuracy: {log_training_results.best_test_acc}")
