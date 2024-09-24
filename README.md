# Zhuang_Bert-BiLSTM-GCN
This repo contains code for 基于预训练多网络融合的低资源壮文文本分类.


## Optimization
- [ x ] 采用
- [ x ] 实现AIMyang/zhuang_bert_mlm 模型，Bi-LSTM和GCN模型的融合
- [ x ] 参数调优，进行训练

## Reference

[*BertGCN*]](https://github.com/ZeroRin/BertGCN)
[*zhuang_bert-Bi-LSTM-GCN*] ([https://github.com/ymcui/Chinese-BERT-wwm](https://huggingface.co/AIMyang/zhuang_bert_mlm))


## Introduction


## Dependencies

Create environment and install required packages for BertGCN using conda:

`conda create --name BertGCN --file requirements.txt -c default -c pytorch -c dglteam -c huggingface`

If the NVIDIA driver version does not support CUDA 10.1 you may edit requirements.txt to use older cudatooklit and the corresponding [dgl](https://www.dgl.ai/pages/start.html) instead.

## Usage

1. Run `python build_graph.py [dataset]` to build the text graph.

2. Run `python finetune_bert.py --dataset [dataset]` 
to finetune the BERT model over target dataset. The model and training logs will be saved to `checkpoint/[bert_init]_[dataset]/` by default. 
Run `python finetune_bert.py -h` to see the full list of hyperparameters.

3. Run `python train_bert_gcn.py --dataset [dataset] --pretrained_bert_ckpt [pretrained_bert_ckpt] -m [m]`
to train the BertGCN. 
`[m]` is the factor balancing BERT and GCN prediction \(lambda in the paper\). 
The model and training logs will be saved to `checkpoint/[bert_init]_[gcn_model]_[dataset]/` by default. 
Run `python train_bert_gcn.py -h` to see the full list of hyperparameters.

Trained BertGCN parameters can be downloaded [here](https://drive.google.com/file/d/1YUl7q34S3pu8KH17yOI68tvcedkrQ39a).

## Acknowledgement

The data preprocess and graph construction are from [TextGCN](https://github.com/yao8839836/text_gcn)
The code are from [BertGCN: Transductive Text Classification by Combining GCN and BERT]
## Citation

```
