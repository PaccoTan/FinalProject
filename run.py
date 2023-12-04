import final
from torch.nn import Softmax as softmax
from model import TCRModel
import torch
out = final.train("a",epochs=10,lr=1e-3)
data = []
mask = []
label = []

for i in range(len(out)):
    data.append(out[i][0])
    mask.append(out[i][1])
    label.append((out[i][2]).reshape((1,1)))
data = torch.stack(data)
mask = torch.stack(mask)
label = torch.cat(label)
soft = softmax(dim=1)
model = TCRModel()
model.load("models/a0.pt")
model.to("cuda")
print(model(data[0:10],mask[0:10]))
print(soft(model(data[0:10],mask[0:10])))
print(label[0:10])
