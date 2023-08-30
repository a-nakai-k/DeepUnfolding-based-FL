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
    E = 5                   # number of epochs
    npseed = 24             # random seed for numpy
elif environment=="B":
    T = 70                  # number of rounds for FL
    npseed = 23             # random seed for numpy
    maxep = 7               # maximum number of local epoch
batch_size = 100            # mini-batch size
mu = 0.01                   # learning rate for ClientUpdate
q = 1                       # parameter for DR-FedAvg
numitr = 10                 # number of fl iterations
alpha = 7                   # parameter for FedAdp
beta = 0.5                  # parameter for FedFa
epsilon = 0.000001          # small constant for FedAdp and FedFa
num_feature = 768           # dimension of hidden layers
outputsize = 120            # dimension of output

# Fixed parameters
weights = torch.load('learnedweights_duwfedavg.pt').T
weights2 = torch.load('learnedweights_duwfednova.pt').T
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
train_X = torch.load('./dogs_alltrainX.pt')
train_y = torch.load('./dogs_alltrainy.pt')
test_X = torch.load('./dogs_alltestX.pt')
test_y = torch.load('./dogs_alltesty.pt')
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

# for FedNova
taus = []
for i in range(K):
    train_loader_node = torch.utils.data.DataLoader(dataset=train_datasets[i], batch_size=batch_size, shuffle=True)
    taus.append(len(train_loader_node)*num_localepochs[i])
sumtau = sum([datasizes_train[i]*taus[i]/N for i in range(K)])

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
class TrainDUW(nn.Module):      # duw-fedavg
    def __init__(self) -> None:
        super(TrainDUW, self).__init__()
        self.thetak = nn.ParameterList([nn.Parameter(weights[i]) for i in range(K)])
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
            if not itr == T:
                sumweights = sum([self.thetak[x][itr]**2 for x in range(K)])
            for node in range(K):
                weight3, bias3 = aveW3, aveb3
                weight3.requires_grad_(True)
                bias3.requires_grad_(True)
                train_loader_node = torch.utils.data.DataLoader(dataset=train_datasets[node], batch_size=batch_size, shuffle=True)
                outputvalues = []
                targetvalues = []
                for ep in range(E):
                    for (input, target) in train_loader_node:
                        input = input.view(len(target),num_feature)
                        output = self.network(weight3,bias3,input)
                        outputvalues.append(output)
                        targetvalues.append(target)
                        if not itr == T:
                            loss = F.nll_loss(output, target)
                            tmploss += loss.item()
                            w3grad = torch.autograd.grad(loss,weight3,retain_graph=True)
                            b3grad = torch.autograd.grad(loss,bias3,retain_graph=True)
                            weight3 = weight3 - mu * w3grad[0].detach()
                            bias3 = bias3 - mu * b3grad[0].detach()
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
    
class TrainDUW2(nn.Module):     # duw-fednova
    def __init__(self) -> None:
        super(TrainDUW2, self).__init__()
        self.thetak = nn.ParameterList([nn.Parameter(weights2[i]) for i in range(K)])
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
            if not itr == T:
                sumweights = sum([self.thetak[x][itr]**2 for x in range(K)])
            for node in range(K):
                weight3, bias3 = aveW3, aveb3
                weight3.requires_grad_(True)
                bias3.requires_grad_(True)
                train_loader_node = torch.utils.data.DataLoader(dataset=train_datasets[node], batch_size=batch_size, shuffle=True)
                outputvalues = []
                targetvalues = []
                for ep in range(E):
                    for (input, target) in train_loader_node:
                        input = input.view(len(target),num_feature)
                        output = self.network(weight3,bias3,input)
                        outputvalues.append(output)
                        targetvalues.append(target)
                        if not itr == T:
                            loss = F.nll_loss(output, target)
                            tmploss += loss.item()
                            w3grad = torch.autograd.grad(loss,weight3,retain_graph=True)
                            b3grad = torch.autograd.grad(loss,bias3,retain_graph=True)
                            weight3 = weight3 - mu * w3grad[0].detach()
                            bias3 = bias3 - mu * b3grad[0].detach()
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

def Gompertz(phi,alpha):                    # for FedAdp
    return alpha * (1 - torch.exp(-torch.exp(-alpha*(phi-1))))

def test_duringtrain(datasets,model):    # for FedFa
    accs_nodes = []
    with torch.no_grad():
        for node in range(K):
            train_loader_node = torch.utils.data.DataLoader(dataset=datasets[node], batch_size=batch_size, shuffle=True)
            correct =  0
            count = 0
            for (input, target) in train_loader_node:
                input = input.view(len(target),num_feature)
                output = model[node](input)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.data).sum()
                count += input.size()[0]
            accs_nodes.append(float(correct)/float(count))
    return accs_nodes

def train_learnedweights(datasets,model,optimizer,loop,method,modelfortheta,iscom,phis_pre):
    running_loss = 0.0
    if method=='dr':
        Fk = [0. for _ in range(K)]
    if method=='adp':
        grads_nodes = []
    if method=='nova' or method=='duwnova':
        taus_t = []
        for node in range(K):
            if int(iscom[loop][node]):
                taus_t.append(taus[node])
            else:
                taus_t.append(0.)
        sumtau_t = sum([datasizes_train[i]*taus_t[i]/N for i in range(K)])
    for node in range(K):
        train_loader_node = torch.utils.data.DataLoader(dataset=datasets[node], batch_size=batch_size, shuffle=True)
        count = 0
        if method=='adp':
            grads_node = []
            if not int(iscom[loop][node]):
                for param in model[node].parameters():
                    grads_node.append(torch.zeros(param.data.view(-1).size()[0]))
        for ep in range(num_localepochs[node]):
            for (input, target) in train_loader_node:
                input = input.view(len(target),num_feature)
                if not method=='nova' or not method=='duwnova':
                    optimizer[node].zero_grad()
                output = model[node](input)
                loss = F.nll_loss(output, target)
                if int(iscom[loop][node]):
                    loss.backward()
                    if method=='adp':
                        countparam = 0
                        for param in model[node].parameters():
                            if count==0:
                                grads_node.append(param.grad.detach().data.view(-1))
                            else:
                                grads_node[countparam] = grads_node[countparam] + param.grad.detach().data.view(-1)
                            countparam += 1
                    if method=='nova' or method=='duwnova':
                        for param in model[node].parameters():
                            param.data = (param - mu*sumtau_t/taus_t[node] * param.grad).detach().requires_grad_()
                    else:
                        optimizer[node].step()
                running_loss += loss.item()
                count += 1
                if method=='dr':
                    Fk[node] += loss.item()
        if method=='adp':
            grads_nodes.append(torch.cat(grads_node))
    # aggregation
    with torch.no_grad():
        if method=='dr':
            sumpkFk = sum([pk[node]*(Fk[node]**(q+1)) for node in range(K)])
        elif method=='adp':
            sumgrads = sum([datasizes_train[i]*grads_nodes[i]/N for i in range(K)])
            phis = []
            sumphis = 0.
            for i in range(K):
                if int(iscom[loop][i]):
                    if np.isnan(torch.acos(torch.inner(sumgrads,grads_nodes[i]) / torch.norm(sumgrads) / torch.norm(grads_nodes[i]))):
                        phis.append(torch.acos(torch.tensor(1. - epsilon)))
                    else:
                        phis.append(torch.acos(torch.inner(sumgrads,grads_nodes[i]) / torch.norm(sumgrads) / torch.norm(grads_nodes[i])))
                    if loop != 0:
                        phis[i] = sum(torch.stack(iscom)[:loop+1,i])*phis_pre[i]/(sum(torch.stack(iscom)[:loop+1,i])+1) + phis[i]/(sum(torch.stack(iscom)[:loop+1,i])+1)
                    sumphis += datasizes_train[i] * torch.exp(Gompertz(phis[i],alpha))
                else:
                    phis.append(torch.tensor(0.))
        elif method=='fa':
            accs_nodes = test_duringtrain(datasets,model)
            sumaccs = 0.
            sumfreqs = 0.
            for node in range(K):
                if int(iscom[loop][node]):
                    sumaccs += accs_nodes[node]
                    sumfreqs += sum(torch.stack(iscom)[:loop+1,node])
            sumAks = 0.
            sumBks = 0.
            for node in range(K):
                if int(iscom[loop][node]):
                    if sumaccs == 0 or accs_nodes[node]/sumaccs == 0:
                        sumAks += -np.log2(epsilon)
                    else:
                        sumAks += -np.log2(accs_nodes[node]/sumaccs)
                    if 1 - sum(torch.stack(iscom)[:loop+1,node])/sumfreqs == 0:
                        sumBks += -np.log2(1 - sum(torch.stack(iscom)[:loop+1,node])/sumfreqs + epsilon)
                    else:
                        sumBks += -np.log2(1 - sum(torch.stack(iscom)[:loop+1,node])/sumfreqs)
            Aks = []
            Bks = []
            for node in range(K):
                if int(iscom[loop][node]):
                    if sum(iscom[loop])==1:
                        Aks.append(1.)
                    elif sumaccs == 0 or accs_nodes[node]/sumaccs == 0:
                        Aks.append(-np.log2(epsilon) / sumAks)
                    else:
                        Aks.append(-np.log2(accs_nodes[node]/sumaccs) / sumAks)
                    if 1 - sum(torch.stack(iscom)[:loop+1,node])/sumfreqs == 0:
                        Bks.append(-np.log2(1 - sum(torch.stack(iscom)[:loop+1,node])/sumfreqs + epsilon) / sumBks)
                    else:
                        Bks.append(-np.log2(1 - sum(torch.stack(iscom)[:loop+1,node])/sumfreqs) / sumBks)
                else:
                    Aks.append(0.)
                    Bks.append(0.)
        ps_global = []
        for params in range(len(list(model[0].parameters()))):
            if method=='du' or method=='duwnova':
                ps_global.append(list(model[0].parameters())[params].data * modelfortheta.thetak[0][loop].data.item())
            elif method=='origin' or method=='nova':
                ps_global.append(list(model[0].parameters())[params].data*len(train_datasets[0])/N)
            elif method=='dr':
                ps_global.append(list(model[0].parameters())[params].data*pk[0]*(Fk[0]**(q+1))/sumpkFk)
            elif method=='adp':
                if int(iscom[loop][0]):
                    ps_global.append(list(model[0].parameters())[params].data * datasizes_train[0]*torch.exp(Gompertz(phis[0],alpha))/sumphis)
                else:
                    ps_global.append(list(model[0].parameters())[params].data * 0)
            elif method=='fa':
                ps_global.append(list(model[0].parameters())[params].data * ((1-beta)*Aks[0] + beta*Bks[0]))
            for node in range(1,K):
                if method=='du' or method=='duwnova':
                    ps_global[params] += list(model[node].parameters())[params].data * modelfortheta.thetak[node][loop].data.item()
                elif method=='origin' or method=='nova':
                    ps_global[params] += list(model[node].parameters())[params].data*len(train_datasets[node])/N
                elif method=='dr':
                    ps_global[params] += list(model[node].parameters())[params].data*pk[node]*(Fk[node]**(q+1))/sumpkFk
                elif method=='adp':
                    if int(iscom[loop][node]):
                        ps_global[params] += list(model[node].parameters())[params].data * datasizes_train[node]*torch.exp(Gompertz(phis[node],alpha))/sumphis
                    else:
                        ps_global[params] += list(model[node].parameters())[params].data * 0
                elif method=='fa':
                    ps_global[params] += list(model[node].parameters())[params].data * ((1-beta)*Aks[node] + beta*Bks[node])
        # parameter sharing
        for node in range(K):
            model[node].l3.weight.data = ps_global[0].clone()
            model[node].l3.bias.data = ps_global[1].clone()
    if method=='adp':
        return running_loss/K, phis
    else:
        return running_loss/K

def test(dataloaders,model):
    correct =  0
    count = 0
    with torch.no_grad():
        for node in range(K):
            for (input, target) in dataloaders[node]:
                input = input.view(len(target),num_feature)
                output = model[node](input)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.data).sum()
                count += input.size()[0]
    return float(correct)/float(count)


#%% Model Sharing
model = Net()       # common initial model
models = model_initialize(model)        # models for proposed DUW-FedAvg
models2 = model_initialize(model)       # models for original FedAvg
models3 = model_initialize(model)       # models for DR-FedAvg
models4 = model_initialize(model)       # models for FedNova
models5 = model_initialize(model)       # models for FedAdp
models6 = model_initialize(model)       # models for FedFa
models7 = model_initialize(model)       # models for proposed DUW-FedNova

# initial model parameters
aveW3 = model.l3.weight.data.clone().requires_grad_(False)
aveb3 = model.l3.bias.data.clone().requires_grad_(False)

print("Model sharing done", flush=True)


#%%
modelDU = TrainDUW()
modelDU2 = TrainDUW2()
for param in modelDU.parameters():
    param.requires_grad = False
    print(param, flush=True)
for param in modelDU2.parameters():
    param.requires_grad = False
    print(param, flush=True)

#%% Federated Learning
accs_DUW = torch.zeros(numitr,T)
accs_DUW2 = torch.zeros(numitr,T)
accs_DR = torch.zeros(numitr,T)
accs_origin = torch.zeros(numitr,T)
accs_nova = torch.zeros(numitr,T)
accs_adp = torch.zeros(numitr,T)
accs_fa = torch.zeros(numitr,T)
losses_DUW = torch.zeros(numitr,T)
losses_DUW2 = torch.zeros(numitr,T)
losses_DR = torch.zeros(numitr,T)
losses_origin = torch.zeros(numitr,T)
losses_nova = torch.zeros(numitr,T)
losses_adp = torch.zeros(numitr,T)
losses_fa = torch.zeros(numitr,T)

for ins in range(numitr):
    print(ins, flush=True)
    iscom = [torch.bernoulli(torch.Tensor(com_prob)) for i in range(T)]
    model = Net()
    models = model_initialize(model)        # models for proposed DUW-FedAvg
    models2 = model_initialize(model)       # models for original FedAvg
    models3 = model_initialize(model)       # models for DR-FedAvg
    models4 = model_initialize(model)       # models for FedNova
    models5 = model_initialize(model)       # models for FedAdp
    models6 = model_initialize(model)       # models for FedFa
    models7 = model_initialize(model)       # models for proposed Fed-Nova

    # DUW-FedAvg
    optimizers = []
    for i in range(K):
        optimizers.append(optim.SGD(models[i].parameters(), lr=mu))
    fl_loss = []
    fl_acc = []
    for loop in range(T):
        # print(loop, flush=True)
        loss = train_learnedweights(train_datasets,models,optimizers,loop,'du',modelDU,iscom,[])
        fl_loss.append(loss)
        acc = test(test_loaders,models)
        fl_acc.append(acc)

    accs_DUW[ins,:] = torch.tensor(fl_acc)
    losses_DUW[ins,:] = torch.tensor(fl_loss)
    print("DUW-FedAvg done", flush=True)

    # original FedAvg
    optimizers2 = []
    for i in range(K):
        optimizers2.append(optim.SGD(models2[i].parameters(), lr=mu))
    fl2_loss = []
    fl2_acc = []
    for loop in range(T):
        # print(loop, flush=True)
        loss = train_learnedweights(train_datasets,models2,optimizers2,loop,'origin',modelDU,iscom,[])
        fl2_loss.append(loss)
        acc = test(test_loaders,models2)
        fl2_acc.append(acc)

    accs_origin[ins,:] = torch.tensor(fl2_acc)
    losses_origin[ins,:] = torch.tensor(fl2_loss)
    print("original FedAvg done", flush=True)

    # DR-FedAvg
    optimizers3 = []
    for i in range(K):
        optimizers3.append(optim.SGD(models3[i].parameters(), lr=mu))
    fl3_loss = []
    fl3_acc = []
    for loop in range(T):
        # print(loop, flush=True)
        loss = train_learnedweights(train_datasets,models3,optimizers3,loop,'dr',modelDU,iscom,[])
        fl3_loss.append(loss)
        acc = test(test_loaders,models3)
        fl3_acc.append(acc)

    accs_DR[ins,:] = torch.tensor(fl3_acc)
    losses_DR[ins,:] = torch.tensor(fl3_loss)
    print("DR-FedAvg done", flush=True)

    # FedNova
    optimizers4 = []
    for i in range(K):
        optimizers4.append(optim.SGD(models4[i].parameters(), lr=mu*sumtau/taus[i]))
    fl4_loss = []
    fl4_acc = []
    for loop in range(T):
        # print(loop, flush=True)
        loss = train_learnedweights(train_datasets,models4,optimizers4,loop,'nova',modelDU,iscom,[])
        fl4_loss.append(loss)
        acc = test(test_loaders,models4)
        fl4_acc.append(acc)

    accs_nova[ins,:] = torch.tensor(fl4_acc)
    losses_nova[ins,:] = torch.tensor(fl4_loss)
    print("FedNova done", flush=True)

    # FedAdp
    optimizers5 = []
    for i in range(K):
        optimizers5.append(optim.SGD(models5[i].parameters(), lr=mu))
    fl5_loss = []
    fl5_acc = []
    for loop in range(T):
        # print(loop, flush=True)
        if loop == 0:
            loss, phis = train_learnedweights(train_datasets,models5,optimizers5,loop,'adp',modelDU,iscom,[])
        else:
            loss, phis = train_learnedweights(train_datasets,models5,optimizers5,loop,'adp',modelDU,iscom,phis)
        fl5_loss.append(loss)
        acc = test(test_loaders,models5)
        fl5_acc.append(acc)

    accs_adp[ins,:] = torch.tensor(fl5_acc)
    losses_adp[ins,:] = torch.tensor(fl5_loss)
    print("FedAdp done", flush=True)

    # FedFa
    optimizers6 = []
    for i in range(K):
        optimizers6.append(optim.SGD(models6[i].parameters(), lr=mu))
    fl6_loss = []
    fl6_acc = []
    for loop in range(T):
        # print(loop, flush=True)
        loss = train_learnedweights(train_datasets,models6,optimizers6,loop,'fa',modelDU,iscom,[])
        fl6_loss.append(loss)
        acc = test(test_loaders,models6)
        fl6_acc.append(acc)

    accs_fa[ins,:] = torch.tensor(fl6_acc)
    losses_fa[ins,:] = torch.tensor(fl6_loss)
    print("FedFa done", flush=True)

    # DUW-FedNova
    optimizers7 = []
    for i in range(K):
        optimizers7.append(optim.SGD(models7[i].parameters(), lr=mu*sumtau/taus[i]))
    fl7_loss = []
    fl7_acc = []
    for loop in range(T):
        # print(loop, flush=True)
        loss = train_learnedweights(train_datasets,models7,optimizers7,loop,'duwnova',modelDU2,iscom,[])
        fl7_loss.append(loss)
        acc = test(test_loaders,models7)
        fl7_acc.append(acc)

    accs_DUW2[ins,:] = torch.tensor(fl7_acc)
    losses_DUW2[ins,:] = torch.tensor(fl7_loss)
    print("DUW-FedNova done", flush=True)

print(torch.mean(losses_DUW, dim=0)[-1], 'plusminus', torch.std(losses_DUW, dim=0)[-1], flush=True)
print(torch.mean(accs_DUW, dim=0)[-1], 'plusminus', torch.std(accs_DUW, dim=0)[-1], flush=True)
print(torch.mean(losses_DUW2, dim=0)[-1], 'plusminus', torch.std(losses_DUW2, dim=0)[-1], flush=True)
print(torch.mean(accs_DUW2, dim=0)[-1], 'plusminus', torch.std(accs_DUW2, dim=0)[-1], flush=True)
print(torch.mean(losses_origin, dim=0)[-1], 'plusminus', torch.std(losses_origin, dim=0)[-1], flush=True)
print(torch.mean(accs_origin, dim=0)[-1], 'plusminus', torch.std(accs_origin, dim=0)[-1], flush=True)
print(torch.mean(losses_DR, dim=0)[-1], 'plusminus', torch.std(losses_DR, dim=0)[-1], flush=True)
print(torch.mean(accs_DR, dim=0)[-1], 'plusminus', torch.std(accs_DR, dim=0)[-1], flush=True)
print(torch.mean(losses_nova, dim=0)[-1], 'plusminus', torch.std(losses_nova, dim=0)[-1], flush=True)
print(torch.mean(accs_nova, dim=0)[-1], 'plusminus', torch.std(accs_nova, dim=0)[-1], flush=True)
print(torch.mean(losses_adp, dim=0)[-1], 'plusminus', torch.std(losses_adp, dim=0)[-1], flush=True)
print(torch.mean(accs_adp, dim=0)[-1], 'plusminus', torch.std(accs_adp, dim=0)[-1], flush=True)
print(torch.mean(losses_fa, dim=0)[-1], 'plusminus', torch.std(losses_fa, dim=0)[-1], flush=True)
print(torch.mean(accs_fa, dim=0)[-1], 'plusminus', torch.std(accs_fa, dim=0)[-1], flush=True)

# %%
