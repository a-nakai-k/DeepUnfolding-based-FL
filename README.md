# DeepUnfolding-based-FL
Python implementation of Deep Unfolding-based Weight FederatedAveraging (DUW-FedAvg).

''Deep unfolding-based weighted averaging for federated learning under heterogeneous environments,'' submitted to ICASSP 2023.


## Requirement
- Pytorch
- matplotlib
- numpy

## Dataset (extracted from MNIST)
| Env. | characteristics | quantity | attached class label | epochs | communication probability | 
| ---- | ---- | ---- | ---- | ---- | ---- |
| I | quantity skew | 1042,1023,862,1184,4459 | all labels | 2 | perfect |
| II | label distribution skew | 6755,6774,6776,6776,6776 | 2,3,5,5,5 | 2 | perfect |
| III | computational skew | 1713,1713,1713,1713,1716 | all labels | 2,1,1,1,1 | perfect |
| IV | communication skew | 1713,1713,1713,1713,1716 | all labels | 2 | 0.2,0.3,0.8,0.9,1 |

- 5clients_env1: Environment I containing quantity skew. Local data are IID but the quantity is not balanced.
- 5clients_env2: Environment II containing label distribution skew. The quantity is balanced but each client only has data for specific labels.
- 5clients_env3_4: Environment III and IV. Local data are IID and balanced.

## Codes
- main_env1_2.py: Code for environment I and II.
- main_env3.py: Code for environment III containing computational capability skew. The number of epochs that can calculate during a round varies across clients.
- main_env4.py: Code for environment IV containing communication capability skew. Each client can transmit the model parameters only with a certain probability.


## License
This project is licensed under the MIT License, see the LICENSE file for details.

## Author
[Ayano NAKAI-KASAI](https://sites.google.com/view/ayano-nakai/home/english)

Graduate School of Engineering, Nagoya Institute of Technology

nakai.ayano@nitech.ac.jp

## Acknowledgment
Datasets are extracted from MNIST by using [this repositry](https://github.com/TsingZ0/PFL-Non-IID).
