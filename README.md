# DeepUnfolding-based-FL
Python implementation of Deep Unfolding-based Weight FederatedAveraging (DUW-FedAvg).

A. Nakai-Kasai and T. Wadayama, ''Deep unfolding-based weighted averaging for federated learning under heterogeneous environments,'' submitting to ICASSP 2023.


## Requirement
- Pytorch
- matplotlib
- numpy

## Dataset (extracted from MNIST)
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
