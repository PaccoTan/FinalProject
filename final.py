from math import ceil
import torch
from model import TCRModel, clf_loss_func
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import random
import matplotlib.pyplot as plt


def load(file,corrupt=False):
    df = pd.read_csv(file)
    tokens = sorted(set(''.join(df['antigen'] + df['TCR'])))
    tMap = {}
    for val, token in enumerate(tokens):
        tMap[token] = val + 1
    if corrupt:
        ant = list(set(df['antigen']))
        tcr = list(set(df['TCR']))
        return ant,tcr,tMap
    data = preprocess(df,tMap)
    labels = torch.Tensor(df["interaction"].values).type(torch.int64)
    return data, labels

def corrupt(seq,tokens,pad,pad2,ratio=0.3):
  data = pad_sequence(to_Tensor(seq,tokens),batch_first=True)
  mask = (data != 0).int()
  a = (torch.randn(data.shape)>ratio).int()
  b = torch.randint(20,data.shape)+1
  labels = (a<=ratio).int()
  out = (data*a + labels*b)*mask
  out = torch.nn.functional.pad(out,(0,pad2-out.shape[-1],0,0)).int()
  labels = torch.nn.functional.pad(labels,(0,pad2-labels.shape[-1]+1,0,0)).float()
  mask = torch.nn.functional.pad(mask, (0, pad2 - mask.shape[-1]+1, 0, 0)).int()
  out = torch.cat([torch.full((out.shape[0],1),pad),out],dim=1)
  return out,labels,mask

def get_masks(data):
    result = data != 0
    return result.int()

def preprocess(df,tokens):
    antigen = to_Tensor(df["antigen"],tokens)
    tcr = to_Tensor(df['TCR'],tokens)
    antigen = pad_sequence(antigen,batch_first=True)
    tcr = pad_sequence(tcr,batch_first=True)
    data = torch.cat([torch.full((len(df), 1), len(tokens) + 1), antigen,
                      torch.full((len(df), 1), len(tokens) + 2), tcr], dim=1).int()
    return data

def to_Tensor(seqs,tokens):
    s = [[*i] for i in seqs]
    return [torch.Tensor([*map(tokens.get, seq)]) for seq in s]

def train(fn,epochs=10,fmodel=None,dev="cuda",lr=1e-3):
    data, labels = load("data.csv")
    data = data.to(dev)
    labels = labels.to(dev)
    masks = get_masks(data).to(dev)
    train = []
    for i in range(data.shape[0]):
        train.append((data[i],masks[i], labels[i]))
    random.shuffle(train)
    t1 = train[0:len(train)//3]
    t2 = train[len(train) // 3:len(train) // 3*2]
    t3 = train[len(train) // 3*2:]
    scores = []
    for k in range(3):
        if fmodel is None:
            model = TCRModel().to(dev)
        else:
            model = TCRModel()
            model.load(fmodel)
            model.to(dev)
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        if k == 0:
            train_set = t1 + t2
        elif k == 1:
            train_set = t1 + t3
        else:
            train_set = t2 + t3
        loader = torch.utils.data.DataLoader(dataset=train_set[:3000], batch_size=128, shuffle=True)
        score = []
        loss2 = torch.nn.CrossEntropyLoss(weight=torch.tensor([1,5]).to(dev))
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     weight_decay=1e-8)
        for epoch in range(epochs):
            if epoch == 25:
                for param in model.parameters():
                    param.requires_grad = True
            loss_epoch = 0
            for batch in loader:
                out = model(batch[0],batch[1])
                optimizer.zero_grad()
                # loss = clf_loss_func(out,batch[2])
                loss = loss2(out, batch[2])
                loss.backward()
                optimizer.step()
                score.append(loss.item())
            scores.append(sum(score)/len(score))
        model.save("models/" + fn + str(k) + ".pt")
        break
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(scores)
    plt.show()
    return train[:3000]
    # with open("file.txt", 'w') as f:
    #     for s in scores:
    #         f.write(str(s) + '\n')
    #     for s in ce:
    #         f.write(str(s) + '\n')

def pretrain(fn,model=None,epochs=10,dev='cuda',ratio=0.3):
    if model is None:
        model = TCRModel().to(dev)
    classifier = Classifier().to(dev)
    ant,tcr,tokens = load("data.csv",corrupt = True)
    max_len = max(map(len,ant+tcr))
    d1, l1, m1 = corrupt(ant,tokens,len(tokens)+1,max_len,ratio)
    d2, l2, m2 = corrupt(tcr,tokens,len(tokens)+2,max_len,ratio)
    data = torch.cat([d1,d2]).to(dev)
    labels = torch.cat([l1,l2]).to(dev)
    masks = torch.cat([m1,m2]).to(dev)
    train = []
    for i in range(data.shape[0]):
        train.append((data[i], masks[i], labels[i]))
    loader = torch.utils.data.DataLoader(dataset=train, batch_size=128, shuffle=True)
    scores = []
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4,
                                 weight_decay=1e-8)
    loss_func = nn.BCELoss()
    for epoch in range(epochs):
        for batch in loader:
            out = model(batch[0], batch[1],classification=False)
            out = out.reshape((out.shape[0] * out.shape[1],out.shape[2]))
            out = classifier(out)
            optimizer.zero_grad()
            loss = loss_func(out, batch[2].reshape((batch[2].shape[0] * batch[2].shape[1],1)))
            loss.backward()
            optimizer.step()
            scores.append(loss.item())
        model.save(fn)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(scores)
    plt.show()

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512,256)
        self.fc2 = nn.Linear(256,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = self.fc(x)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    pass