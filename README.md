# ID-GRec
**ID-based Graph Recommendation Framework (PyTorch)**
<p float="left"><img src="https://img.shields.io/badge/Python-v3.8.18-blue"> <img src="https://img.shields.io/badge/PyTorch-v2.1.0-green"> <br>

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
| BPRMF          | BPR: Bayesian Personalized Ranking from Implicit Feedback [[Paper](https://arxiv.org/abs/1205.2618)]                                              | UAI'09          | base     |
| GC-MC          | Graph Convolutional Matrix Completion [[Paper](https://arxiv.org/abs/1706.02263)]                                                                 | KDD'17          | GNN      |
| LR-GCCF        | Revisiting Graph based Collaborative Filtering: A Linear Residual Graph Convolutional Network Approach [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5330)]| AAAI'20         | GNN      |
| NGCF           | Neural graph collaborative filtering [[Paper](https://dl.acm.org/doi/abs/10.1145/3331184.3331267)]                                                                   | SIGIR'19        | GNN      |
| LightGCN       | LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation [[Paper](https://dl.acm.org/doi/abs/10.1145/3397271.3401063)]                        | SIGIR'20        | GNN      |
| IMP-GCN        | Interest-aware Message-passing GCN for Recommendation [[Paper](https://dl.acm.org/doi/abs/10.1145/3442381.3449986)]                                                  | WWW'21          | GNN      |
| SGL            | Self-supervised Graph Learning for Recommendation [[Paper](https://dl.acm.org/doi/abs/10.1145/3404835.3462862)]                                                      | SIGIR'21        | GNN+SSL  |
| CVGA           | Revisiting Graph-based Recommender Systems from the Perspective of Variational Auto-Encoder [[Paper](https://dl.acm.org/doi/full/10.1145/3573385)]           | TOIS'22         | GNN+SSL  |
| SimGCL         | Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation [[Paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531937)]               | SIGIR'22        | GNN+SSL  |
| DirectAU       | Towards Representation Alignment and Uniformity in Collaborative Filtering [[Paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539253)]                             | KDD'22          | (GNN)+SSL|
| NCL            | Improving graph collaborative filtering with neighborhood-enriched contrastive learning [[Paper](https://dl.acm.org/doi/abs/10.1145/3485447.3512104)]               | WWW'22          | GNN+SSL  |
| HCCF           | Hypergraph Contrastive Collaborative Filtering [[Paper](https://dl.acm.org/doi/abs/10.1145/3477495.3532058)]                                                        | SIGIR'22        | GNN+SSL  |
| XSimGCL        | XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation [[Paper](https://ieeexplore.ieee.org/abstract/document/10158930/)]                       | TKDE'23         | GNN+SSL  |
| LightGCL       | LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation [[Paper](https://openreview.net/forum?id=FKXVK9dyMM)]                           | ICLR'23         | GNN+SSL  |
| DCCF           | Disentangled Contrastive Collaborative Filtering [[Paper](https://dl.acm.org/doi/abs/10.1145/3539618.3591665)]                                                      | SIGIR'23        | GNN+SSL  |
| CGCL           | Candidate–aware Graph Contrastive Learning for Recommendation [[Paper](https://dl.acm.org/doi/10.1145/3539618.3591647)]                                          | SIGIR'23        | GNN+SSL  |
| MAWU        | Toward a Better Understanding of Loss Functions for Collaborative Filtering [[Paper](https://dl.acm.org/doi/10.1145/3583780.3615086)]                                                | CIKM'23         | GNN+SSL  |
| RecDCL         | RecDCL: Dual Contrastive Learning for Recommendation [[Paper](https://dl.acm.org/doi/abs/10.1145/3589334.3645533)]                                                  | WWW'24          | GNN+SSL  |
| BIGCF          | Exploring the Individuality and Collectivity of Intents behind Interactions for Graph Collaborative Filtering [[Paper](https://dl.acm.org/doi/abs/10.1145/3626772.3657738)]  | SIGIR'24        | GNN+SSL  |
| SCCF           | Unifying Graph Convolution and Contrastive Learning in Collaborative Filtering [[Paper](https://dl.acm.org/doi/abs/10.1145/3637528.3671840)]                                 | KDD'24          | (GNN)+SSL  |
| LightGCN++     | Revisiting LightGCN: Unexpected Inflexibility, Inconsistency, and A Remedy Towards Improved Recommendation [[Paper](https://dl.acm.org/doi/abs/10.1145/3640457.3688176)]     | RecSys'24       | GNN      |
| LightGODE      | Do We Really Need Graph Convolution During Training? Light Post-Training Graph-ODE for Efficient Recommendation [[Paper](https://dl.acm.org/doi/abs/10.1145/3627673.3679773)]| CIKM'24         | GNN      |
| EGCF           | Simplify to the Limit! Embedding-less Graph Collaborative Filtering for Recommender Systems [[Paper](https://dl.acm.org/doi/10.1145/3701230)]                                | TOIS'24         | GNN+SSL  |
| MixRec         | MixRec: Individual and Collective Mixing Empowers Data Augmentation for Recommender Systems [[Paper](https://arxiv.org/abs/2501.13579)]                                      | WWW'25          | (GNN)+SSL  |
| LightCCF       | Unveiling Contrastive Learning’s Capability of Neighborhood Aggregation for Collaborative Filtering [[Paper](https://arxiv.org/abs/2504.10113)]                              | SIGIR'25          | (GNN)+SSL  |

## Configuration File Description
The folder named “configure” contains the hyperparameters, datasets, and other miscellaneous settings for all implemented methods. Except for model-specific hyperparameters, the settings listed at the top are common to all models. Using LightGCN.txt as an example, the following provides an introduction to the common settings:
1. `dataset_path` = ./dataset/`: Specifies the file path to the folder containing the dataset files.
2. `dataset = yelp2018`: Sets the name of the dataset to be used, in this case, "yelp2018."
3. `top_K = [5, 10, 20]`: Defines the list of top-K values for evaluation metrics, meaning the model will evaluate its performance using the top 5, top 10, and top 20 predictions.
4. `training_epochs = 1000`: Specifies the number of training epochs, or iterations over the entire dataset, set to 1000.
5. `early_stopping = 10`: Sets the early stopping criterion; training will halt if there's no improvement in performance for 10 consecutive tests.
6. `embedding_size = 64`: Defines the size of the embeddings (or the dimensionality of the embedding vectors) for the model, set to 64.
7. `batch_size = 2048`: Sets the number of samples in each batch for training, with 2048 samples per batch.
8. `test_batch_size = 200`: Specifies the number of samples in each batch for testing, set to 200.
9. `learn_rate = 0.001`: Sets the learning rate, which controls the step size during optimization, to 0.001.
10. `reg_lambda = 0.0001`: Specifies the regularization parameter (often for weight decay), which helps prevent overfitting, set to 0.0001.
11. `GCN_layer = 3`: Defines the number of graph convolutional network (GCN) layers in the model, set to 3.
12. `sparsity_test = 0`: Indicates whether a sparsity test is performed. Setting this to 0 generally means the sparsity test is disabled. If you want to perform a sparsity test, please set it to 1.

## Basic Comparisons
Taking the Yelp2018 dataset provided in the LightGCN paper as an example, the following table presents the reproduced results from ID-GRec with the results reported in the original paper (all publications that used the Yelp2018 dataset):
| **Model Name** | **Recall@20 (paper)** | **Recall@20 (ID-GRec)** | **NDCG@20 (paper)** | **NDCG@20 (ID-GRec)** |
|----------------|---------------|-------------|---------------|-------------|
| BPRMF          |        -      |   0.0554    |       -       |    0.0453   |
| NGCF           |     0.0579    |   0.0573    |    0.0477     |    0.0465   |
| LightGCN       |     0.0639    |   0.0641    |    0.0525     |    0.0527   |
| SGL-ED         |     0.0675    |   0.0675    |    0.0555     |    0.0555   |
| CVGA           |     0.0694    |   0.0691    |    0.0571     |    0.0570   |
| SimGCL         |     0.0721    |   0.0722    |    0.0601     |    0.0599   |
| XSimGCL        |     0.0723    |   0.0724    |    0.0604     |    0.0599   |
| EGCF           |     0.0748    |   0.0749    |    0.0617     |    0.0619   |

## Acknowledgement

ID-GRec is based on numerous outstanding existing works. We have mainly drawn on the following open-source frameworks and would like to express our sincere gratitude for their contributions:
> LightGCN: https://github.com/gusye1234/LightGCN-PyTorch and https://github.com/kuandeng/LightGCN (Training process, testing metric calculations, and original datasets)

> NGCF: https://github.com/xiangwang1223/neural_graph_collaborative_filtering (Sparsity testing process)

> SelfRec: https://github.com/Coder-Yu/SELFRec (Framework design process)

> SSLRec: https://github.com/HKUDS/SSLRec (Output log)

All the implemented methods in ID-GRec have been reproduced and integrated based on the source code provided by the original authors. Due to limitations in time and personal capabilities, there may be errors in the implementation of some models. We sincerely apologize if this causes any inconvenience to your research. If you have any questions regarding ID-GRec, please contact zhangyi.ahu@gmail.com.

## Citation
If you find this work is helpful to your research, please consider citing our paper:
```
@article{zhang2024simplify,
  title={Simplify to the Limit! Embedding-less Graph Collaborative Filtering for Recommender Systems},
  author={Zhang, Yi and Zhang, Yiwen and Sang, Lei and Sheng, Victor S},
  journal={ACM Transactions on Information Systems},
  year={2024},
  publisher={ACM New York, NY}
}
```
