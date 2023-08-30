# DeepUnfolding-based-FL
Python implementation of Deep Unfolding-based Weight Federated Averaging (DUW-FedAvg) / DUW-FedNova.

''Deep unfolding-based weighted averaging for federated learning in heterogeneous environments,'' submitted to IEEE Open Journal of Signal Processing.


## Requirements
- Pytorch
- matplotlib
- numpy
- pandas
- transformers
- scikit-learn

## Datasets
### MNIST for Fig. 2
- mnist/2clients: IID but unbalanced in data quantity. One client has 1760 training data and another has 5739 data.

### MNIST for Fig. 3
| Env. | characteristics | quantity | attached class label | epochs | communication probability | 
| ---- | ---- | ---- | ---- | ---- | ---- |
| I | quantity skew | 1042,1023,862,1184,4459 | all labels | 2 | perfect |
| II | label distribution skew | 6755,6774,6776,6776,6776 | 2,3,5,5,5 | 2 | perfect |
| III | computational skew | 1713,1713,1713,1713,1716 | all labels | 2,1,1,1,1 | perfect |
| IV | communication skew | 1713,1713,1713,1713,1716 | all labels | 2 | 0.2,0.3,0.8,0.9,1 |

- mnist/5clients_env1: Environment I containing quantity skew. Local data are IID but the quantity is not balanced.
- mnist/5clients_env2: Environment II containing label distribution skew. The quantity is balanced but each client only has data for specific labels.
- mnist/5clients_env3_4: Environment III and IV. Local data are IID and balanced.

### Dogs for Table 3
| Property | client 0 | client 1| client 2 | client 3 | client 4 | 
| ---- | ---- | ---- | ---- | ---- | ---- |
| #train data | 3295 | 3345 | 1868 | 3178| 4773 |
| #class labels | 24 | 25 | 18 | 25 | 39 |

- dogs/2clients: Non-IID datasets using the Dirichlet distribution. The concentration parameter of the Dirichlet distribution was set to 0.0005.


## Model
- 3 fully connected layers for MNIST datasets
- Pretrained ViT + 1 fully connected layer for Stanford Dogs datasets


## Codes
### for Fig. 2
- main_mnist_rate.py: Code for calculating mean of the variance of learned weights.

### for Fig. 3
- main_mnist_env1_2.py: Code for environment I and II.
- main_mnist_env3.py: Code for environment III containing computational capability skew. The number of epochs that can calculate during a round varies across clients.
- main_mnist_env4.py: Code for environment IV containing communication capability skew. Each client can transmit the model parameters only with a certain probability.

### for Table 3
1. preprocess_dogs.py: Preparing ViT outputs.
2. main_dogs_training_duwfedavg/duwfednova.py: Performing the proposed preprocessing for FedAvg or FedNova to obtain the optimized aggregation weights.
3. main_dogs_evaluation.py: Performing federated learning and evaluating the obtained model.


## License
This project is licensed under the MIT License, see the LICENSE file for details.

## Author
[Ayano NAKAI-KASAI](https://sites.google.com/view/ayano-nakai/home/english)

Graduate School of Engineering, Nagoya Institute of Technology

nakai.ayano@nitech.ac.jp

## Acknowledgment
Datasets are extracted from MNIST by using [this repositry](https://github.com/TsingZ0/PFL-Non-IID) and its extension.
