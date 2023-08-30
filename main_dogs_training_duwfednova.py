#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

#%% Parameters
environment = "A"           # A: Environment A, B: Environment B
if environment=="A":
    T = 100                 # number of rounds for FL
    M = 5000                # number of learning iterations
    E = 5                   # number of epochs
    npseed = 24             # random seed for numpy
elif environment=="B":
    T = 70                  # number of rounds for FL
    M = 2000                # number of learning iterations
    npseed = 23             # random seed for numpy
    maxep = 7               # maximum number of local epoch
batch_size = 100            # mini-batch size
lr_du = 0.0005              # learning rate for proposed preprocess
mu = 0.01                   # learning rate for ClientUpdate
num_feature = 768           # dimension of hidden layers
outputsize = 120            # dimension of output

# Fixed parameters
K = 5                   # number of clients
datalocation = './dogs/5clients/'
datasizes_train = [3295, 3345, 1868, 3178, 4773]     # number of all training data
N = sum(datasizes_train)
if environment=="A":
    num_localepochs = [E for _ in range(K)]
elif environment=="B":
    np.random.seed(seed=npseed)
    num_localepochs = np.random.randint(maxep, size=K) + 1
np.random.seed(seed=npseed)
com_prob = np.random.rand(K)
print('num_localepochs = ', num_localepochs, flush=True)
print('com_prob = ', com_prob, flush=True)
print("Set parameters", flush=True)

#%% Global Data Load
train_X = torch.load('dogs_alltrainX.pt')
train_y = torch.load('dogs_alltrainy.pt')
test_X = torch.load('dogs_alltestX.pt')
test_y = torch.load('dogs_alltesty.pt')
test_set = torch.utils.data.TensorDataset(test_X,test_y)

print("Download done", flush=True)

#%% Local Data
datasizes_test = []
for i in range(K):
    datasizes_test.append(int(round(len(test_set)/K)))
test_sets = torch.utils.data.random_split(dataset=test_set, lengths=datasizes_test, generator=torch.Generator().manual_seed(42))

test_loaders = []
train_datasets = []

for node in range (K):
    localtraindata = np.load(datalocation+'train/'+str(node)+'.npz', allow_pickle=True)     # training data load
    localtraindata = np.atleast_1d(localtraindata['data'])
    inputs = localtraindata[0]['x']
    targets = localtraindata[0]['y']
    tensor_X = torch.stack([torch.from_numpy(i) for i in inputs])
    tensor_y = torch.stack([torch.from_numpy(np.array(i)) for i in targets])
    train_datasets.append(torch.utils.data.TensorDataset(tensor_X,tensor_y))
    test_loaders.append(torch.utils.data.DataLoader(dataset=test_sets[node], batch_size=batch_size, shuffle=False))
pk = [len(train_datasets[node])/N for node in range(K)]

print("Set local data", flush=True)


#%% Definition of Initial Network
class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__() 
        self.l3 = nn.Linear(num_feature, outputsize)
    def forward(self, x):
        x = self.l3(x)
        return F.log_softmax(x, dim=1)


#%% Definition of Unfolded FL
class TrainDUW(nn.Module):
    def __init__(self) -> None:
        super(TrainDUW, self).__init__()
        self.thetak = nn.ParameterList([nn.Parameter(torch.ones(T)*np.sqrt(len(train_datasets[i])/N)) for i in range(K)])
        # initial value: N_k/N
    def network(self, W3, b3, x):
        x = F.log_softmax(torch.matmul(x,W3.T)+b3.T, dim=1)
        return x
    def forward(self, aveW3, aveb3):
        outputlists = []
        targetlists = []
        for itr in range(T+1):
            tmploss = 0.
            outputlist = []
            targetlist = []
            iscom = torch.bernoulli(torch.Tensor(com_prob))
            if not itr == T:
                sumweights = sum([self.thetak[x][itr]**2 for x in range(K)])
            taus_t = []
            for node in range(K):
                if int(iscom[node]):
                    taus_t.append(taus[node])
                else:
                    taus_t.append(0.)
            sumtau_t = sum([datasizes_train[i]*taus_t[i]/N for i in range(K)])
            for node in range(K):
                weight3, bias3 = aveW3, aveb3
                weight3.requires_grad_(True)
                bias3.requires_grad_(True)
                train_loader_node = torch.utils.data.DataLoader(dataset=train_datasets[node], batch_size=batch_size, shuffle=True)
                outputvalues = []
                targetvalues = []
                for ep in range(num_localepochs[node]):
                    for (input, target) in train_loader_node:
                        input = input.view(len(target),num_feature)
                        output = self.network(weight3,bias3,input)
                        outputvalues.append(output)
                        targetvalues.append(target)
                        if int(iscom[node]):
                            if not itr == T:
                                loss = F.nll_loss(output, target)
                                tmploss += loss.item()
                                w3grad = torch.autograd.grad(loss,weight3,retain_graph=True)
                                b3grad = torch.autograd.grad(loss,bias3,retain_graph=True)
                                weight3 = weight3 - mu*sumtau_t/taus_t[node] * w3grad[0].detach()
                                bias3 = bias3 - mu*sumtau_t/taus_t[node] * b3grad[0].detach()
                if not itr == T:
                    if node == 0:
                        weight0 = self.thetak[node][itr]**2 / sumweights
                        W3, b3 = weight3*weight0, bias3*weight0
                    else:
                        weightnode = self.thetak[node][itr]**2 / sumweights
                        W3, b3 = W3 + weight3*weightnode, b3 + bias3*weightnode
                outputlist.append(outputvalues)
                targetlist.append(targetvalues)
            aveW3, aveb3 = W3, b3
            outputlists.append(outputlist)
            targetlists.append(targetlist)
        return outputlists, targetlists

print("Network defined", flush=True)


#%% Functions
def model_initialize(initmodel):
    models = []
    for i in range(K):
        models.append(Net())
    for node in range(K):
        models[node].l3.weight.data = initmodel.l3.weight.data.clone()
        models[node].l3.bias.data = initmodel.l3.bias.data.clone()
    return models


#%% Model Sharing
model = Net()       # common initial model
models = model_initialize(model)        # models for proposed DUW+FedNova

# initial model parameters
aveW3 = model.l3.weight.data.clone().requires_grad_(False)
aveb3 = model.l3.bias.data.clone().requires_grad_(False)

print("Model sharing done", flush=True)


#%%
modelDU = TrainDUW()
learnedweights = torch.zeros(M+1, T, K)
sumweights = torch.zeros(T)
for itr in range(T):
    sumweights[itr] = sum([modelDU.thetak[x][itr].item()**2 for x in range(K)])
for node in range(K):
    learnedweights[0,:,node] = modelDU.thetak[node].detach()**2 / sumweights
optimizerDU = optim.Adam(modelDU.parameters(), lr=lr_du)

#%%
# Training of Deep Unfolding-based Weights
outerloss = []
i = 0
for loop in range(M):
    i = i+1
    print(i, flush=True)
    optimizerDU.zero_grad()
    outputlists, targetlists = modelDU(aveW3, aveb3)
    loss = 0
    for j in range(T+1):
        for node in range(K):
            num_localdata = len(outputlists[j][node])
            for l in range(num_localdata):
                loss += F.nll_loss(outputlists[j][node][l], targetlists[j][node][l])
    loss.backward()
    optimizerDU.step()
    outerloss.append(loss.item())
    sumweights = torch.zeros(T)
    for itr in range(T):
        sumweights[itr] = sum([modelDU.thetak[x][itr].item()**2 for x in range(K)])
    for node in range(K):
        learnedweights[i,:,node] = modelDU.thetak[node].detach()**2 / sumweights
    if loop%100==99:
        fig = plt.figure()
        plt.plot(outerloss)
        plt.xlabel("iteration m", fontsize=16)
        plt.ylabel("loss", fontsize=16)
        plt.tick_params(labelsize=16)
        plt.tight_layout()
        fig.savefig("outerloss_du.png")
        fig2 = plt.figure()
        for node in range(K):
            labelname = 'learned theta ' + str(node)
            plt.plot([i for i in learnedweights[loop,:,node]], label=labelname)
        plt.legend(fontsize=18)
        plt.xlabel("round t", fontsize=16)
        plt.ylabel("learned theta", fontsize=16)
        plt.tick_params(labelsize=16)
        plt.tight_layout()
        fig2.savefig("learned_thetak.png")
        print('Learned theta: ', learnedweights[loop,:,:], flush=True)
        fig3 = plt.figure()
        for itr in range(T):
            labelname = 'round ' + str(itr)
            plt.plot([i for i in learnedweights[:loop,itr,4]], label=labelname)     # trajectory of client 0's weight for each round
        plt.legend()
        plt.xlabel("iteration m", fontsize=16)
        plt.ylabel("theta4", fontsize=16)
        plt.tick_params(labelsize=16)
        plt.tight_layout()
        fig3.savefig("theta4.png")

print("Deep unfolding done", flush=True)
# print ('outerloss = ', outerloss, flush=True)

#%% Figures
# loss during deep unfolding
fig = plt.figure()
plt.plot(outerloss)
plt.xlabel("iteration m", fontsize=16)
plt.ylabel("loss", fontsize=16)
plt.tick_params(labelsize=16)
plt.tight_layout()
fig.savefig("outerloss_du.png")

# learned thetak
fig2 = plt.figure()
for node in range(K):
    labelname = 'learned theta ' + str(node)
    plt.plot([i for i in learnedweights[M,:,node]], label=labelname)
plt.legend(fontsize=18)
plt.xlabel("round t", fontsize=16)
plt.ylabel("learned theta", fontsize=16)
plt.tick_params(labelsize=16)
plt.tight_layout()
fig2.savefig("learned_thetak.png")
print('Learned theta: ', learnedweights[M,:,:], flush=True)

# trajectory of weight during deep unfolding
fig3 = plt.figure()
for itr in range(T):
    labelname = 'round ' + str(itr)
    plt.plot([i for i in learnedweights[:,itr,4]], label=labelname)     # trajectory of client 2's weight for each round
plt.legend()
plt.xlabel("iteration m", fontsize=16)
plt.ylabel("theta4", fontsize=16)
plt.tick_params(labelsize=16)
plt.tight_layout()
fig3.savefig("theta4.png")

torch.save(learnedweights[M,:,:], 'learnedweights_duwfednova.pt')
