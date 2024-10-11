# ID-GRec
**ID-based Graph Recommendation Framework (PyTorch)**

ID-GRec is a **graph recommendation framework** based on Python and Pytorch, which contains the current mainstream and latest graph recommendation methods, the most classic datasets, evaluation metrics and testing processes. One of the advantages of ID-GRec is that it is easy to get started, with a simple configuration of the model parameters, and the training can be executed in a simple one-step commands without any other operations. 

ID-GRec is dedicated to exploring the latest research results on graph-based recommender systems and comparing them in a unified framework. Therefore, we adopt LightGCN, a classic work on graph recommender systems, as a benchmark in ID-GRec and refer to a large number of related works and frameworks. Further, we roughly categorize the existing methods into *graph neural network-based methods (GNN)* and *graph self-supervised learning-based methods (SSL)*. And they are compared under a unified framework, which includes publications in various flagship conferences and journals in recent years. In addition, we also simply implement BPRMF as a most basic comparison baseline as well as an extensible template.

## Environment (based on our test platform)
```
python == 3.8.18
pytorch == 2.1.0 (cuda:12.1)
scipy == 1.10.1
numpy == 1.24.3
tdqm == 4.65.0
```
> For some special models, additional third-party libraries may be required.

## Examples to Run 
Steps to run the code (MODEL_NAME is the name of the model):
1. In the folder . /configure to configure the MODEL_NAME.txt file;
2. Run main.py `python main.py` and select the identifier of MODEL_NAME or specify through the command line:`python main.py --model=MODEL_NAME`

Example:
If you want to run LightGCN:
1. In the folder . /configure to configure the LightGCN.txt file;
2. Run main.py `python main.py` and select the identifier of LightGCN or specify through the command line:`python main.py --model=LightGCN`

## Implemented Model List

| **Model Name** | **Paper**                                                                                              | **Publication** | **Type** |
|----------------|--------------------------------------------------------------------------------------------------------|-----------------|----------|
| BPRMF          | BPR: Bayesian personalized ranking from implicit feedback                                              | UAI'09          | base     |
| GC-MC          | Graph convolutional matrix completion                                                                  | KDD'17          | GNN      |
| LR-GCCF        | Revisiting graph based collaborative filtering: A linear residual graph convolutional network approach | AAAI'20         | GNN      |
| NGCF           | Neural graph collaborative filtering                                                                   | SIGIR'19        | GNN      |
| LightGCN       | LightGCN: Simplifying and powering graph convolution network for recommendation                        | SIGIR'20        | GNN      |
| IMP-GCN        | Interest-aware message-passing GCN for recommendation                                                  | WWW'21          | GNN      |
| SGL            | Self-supervised graph learning for recommendation                                                      | SIGIR'21        | GNN+SSL  |
| CVGA           | Revisiting Graph-based Recommender Systems from the Perspective of Variational Auto-Encoder            | TOIS'22         | GNN+SSL  |
| SimGCL         | Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation                | SIGIR'22        | GNN+SSL  |
| DirectAU       | Towards Representation Alignment and Uniformity in Collaborative Filtering                             | KDD'22          | (GNN)+SSL|
| NCL            | Improving graph collaborative filtering with neighborhood-enriched contrastive learning                | WWW'22          | GNN+SSL  |
| HCCF           | Hypergraph Contrastive Collaborative Filtering                                                         | SIGIR'22        | GNN+SSL  |
| XSimGCL        | XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation                        | TKDE'23         | GNN+SSL  |
| LightGCL       | LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation                           | ICLR'23         | GNN+SSL  |
| VGCL           | Generative-Contrastive Graph Learning for Recommendation                                               | SIGIR'23        | GNN+SSL  |
| DCCF           | Disentangled Contrastive Collaborative Filtering                                                       | SIGIR'23        | GNN+SSL  |
| CGCL           | Candidate–aware Graph Contrastive Learning for Recommendation                                          | SIGIR'23        | GNN+SSL  |
| GraphAU        | Graph-based Alignment and Uniformity for Recommendation                                                | CIKM'23         | GNN+SSL  |
| RecDCL         | RecDCL: Dual Contrastive Learning for Recommendation                                                   | WWW'24          | GNN+SSL  |
| BIGCF          | Exploring the Individuality and Collectivity of Intents behind Interactions for Graph Collaborative Filtering  | SIGIR'24        | GNN+SSL  |
| SCCF           | Unifying Graph Convolution and Contrastive Learning in Collaborative Filtering                         | KDD'24          | GNN+SSL  |
| EGCF           | Simplify to the Limit! Embedding-less Graph Collaborative Filtering for Recommender Systems            | TOIS'24         | GNN+SSL  |
## Basic Comparisons
Taking the Yelp2018 dataset provided in the LightGCN paper as an example, the following table presents the reproduced results from ID-GRec with the results reported in the original paper (all publications that used the Yelp2018 dataset):
| **Model Name** | **Recall@20 (paper)** | **Recall@20 (ID-GRec)** | **NDCG@20 (paper)** | **NDCG@20 (ID-GRec)** |
|----------------|---------------|-------------|---------------|-------------|
| BPRMF          |        -      |   0.0554    |       -       |    0.0453   |
| NGCF           |     0.0579    |   0.0573    |    0.0477     |    0.0465   |
| LightGCN       |     0.0639    |   0.0641    |    0.0525     |    0.0527   |
| SGL            |     0.0675    |   0.0675    |    0.0555     |    0.0555   |
| CVGA           |     0.0694    |   0.0691    |    0.0571     |    0.0570   |
| SimGCL         |     0.0721    |   0.0722    |    0.0601     |    0.0599   |
| XSimGCL        |     0.0723    |   0.0724    |    0.0604     |    0.0599   |

