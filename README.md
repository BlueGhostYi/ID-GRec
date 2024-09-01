# ID-GRec
**ID-based Graph Recommendation Framework (PyTorch)**

ID-GRec is a **graph recommendation framework** based on Python and Pytorch, which contains the current mainstream and latest graph recommendation methods, the most classic datasets, evaluation metrics and testing processes. One of the advantages of ID-GRec is that it is easy to get started, with a simple configuration of the model parameters, and the training can be executed in a simple one-step commands without any other operations. 

ID-GRec is dedicated to exploring the latest research results on graph-based recommender systems and comparing them in a unified framework. Therefore, we adopt LightGCN, a classic work on graph recommender systems, as a benchmark in ID-GRec and refer to a large number of related works and frameworks. Further, we roughly categorize the existing methods into *graph neural network-based methods (GNN)* and *graph self-supervised learning-based methods (SSL)*. And they are compared under a unified framework, which includes publications in various flagship conferences and journals in recent years. In addition, we also simply implement BPRMF as a most basic comparison baseline as well as an extensible template.

## Environment (based on our test platform)
```
python == 3.8.18
pytorch == 2.1.0 (cuda:12.1)
torch-sparse == 0.6.18
scipy == 1.10.1
numpy == 1.24.3
tdqm == 4.65.0
```
> For some special models, additional third-party libraries may be required.

## Examples to Run 
Steps to run the code (MODEL_NAME is the name of the model):
1. In the folder . /configure to configure the MODEL_NAME.txt file;
2. Run main.py and select the MODEL_NAME.

## Implemented Model List

| **Model Name** | **Paper**                                                                                              | **Publication** | **Type** |
|----------------|--------------------------------------------------------------------------------------------------------|-----------------|----------|
| BPRMF          | BPR: Bayesian personalized ranking from implicit feedback                                              | UAI'09          | base     |
| GC-MC          | Graph convolutional matrix completion                                                                  | 17              | GNN      |
| LR-GCCF        | Revisiting graph based collaborative filtering: A linear residual graph convolutional network approach | AAAI'20         | GNN      |
| NGCF           | Neural graph collaborative filtering                                                                   | SIGIR'19        | GNN      |
| LightGCN       | LightGCN: Simplifying and powering graph convolution network for recommendation                        | SIGIR'20        | GNN      |
| IMP-GCN        | Interest-aware message-passing GCN for recommendation                                                  | WWW'21          | GNN      |
| SGL            | Self-supervised graph learning for recommendation                                                      | SIGIR'21        | GNN+SSL  |
| CVGA           | Revisiting Graph-based Recommender Systems from the Perspective of Variational Auto-Encoder            | TOIS'22         | GNN+VAE  |
| SimGCL         | Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation                | SIGIR'22        | GNN+SSL  |
| DirectAU       | Towards Representation Alignment and Uniformity in Collaborative Filtering                             | KDD'22          | (GNN)+SSL|
| NCL            |                                                                                                        | WWW'22          | GNN+SSL  |
| HCCF           |                                                                                                        | SIGIR'22        | GNN+SSL  |
| XSimGCL        |                                                                                                        | TKDE'23         | GNN+SSL  |
| LightGCL       |                                                                                                        | ICLR'23         | GNN+SSL  |
| VGCL           |                                                                                                        | SIGIR'23        | GNN+SSL  |
| DCCF           |                                                                                                        | SIGIR'23        | GNN+SSL  |
| CGCL           |                                                                                                        | SIGIR'23        | GNN+SSL  |
| GraphAU        |                                                                                                        | CIKM'23         | GNN+SSL  |
| AU+            |                                                                                                        | AAAI'24         | GNN+SSL  |
| RecDCL         |                                                                                                        | WWW'24          | GNN+SSL  |
| BIGCF          |                                                                                                        | SIGIR'24        | GNN+SSL  |
| SCCF           |                                                                                                        | KDD'24          | GNN+SSL  |




