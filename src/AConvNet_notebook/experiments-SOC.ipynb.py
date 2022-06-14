import matplotlib.pyplot as plt

import numpy as np

import json
import glob
import sys
sys.path.append('../')
import os

from tqdm import tqdm
import torchvision
import torch

from AConvNet_utils import common
from AConvNet_data import preprocess
from AConvNet_data import loader
import AConvNet_model

from sklearn import metrics
from AConvNet_data import mstar

import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(path, is_train, name, batch_size):

    _dataset = loader.Dataset(
        path, name=name, is_train=is_train,
        transform=torchvision.transforms.Compose([
            preprocess.CenterCrop(88), torchvision.transforms.ToTensor()
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        _dataset, batch_size=batch_size, shuffle=is_train, num_workers=1
    )
    return data_loader


def evaluate(_m, ds):
    
    num_data = 0
    corrects = 0
    
    _m.net.eval()
    _softmax = torch.nn.Softmax(dim=1)
    for i, data in enumerate(ds):
        images, labels, _ = data

        predictions = _m.inference(images)
        predictions = _softmax(predictions)

        _, predictions = torch.max(predictions.data, 1)
        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)
        corrects += (predictions == labels.to(m.device)).sum().item()

    accuracy = 100 * corrects / num_data
    return accuracy

def confusion_matrix(_m, ds):
    _pred = []
    _gt = []
    
    _m.net.eval()
    _softmax = torch.nn.Softmax(dim=1)
    for i, data in enumerate(ds):
        images, labels, _ = data
        
        predictions = _m.inference(images)
        predictions = _softmax(predictions)

        _, predictions = torch.max(predictions.data, 1)
        labels = labels.type(torch.LongTensor)
        
        _pred += predictions.cpu().tolist()
        _gt += labels.cpu().tolist()
        
    conf_mat = metrics.confusion_matrix(_gt, _pred)
        
    return conf_mat

with open('../AConvNet_MSTAR_experiments/history/history-AConvNet-SOC_2_channels.json') as f:
    history = json.load(f)

training_loss = history['loss']
test_accuracy = history['accuracy']

epochs = np.arange(len(training_loss))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

plot1, = ax1.plot(epochs, training_loss, marker='.', c='blue', label='loss')
plot2, = ax2.plot(epochs, test_accuracy, marker='.', c='red', label='accuracy')
plt.legend([plot1, plot2], ['loss', 'accuracy'], loc='upper right')

plt.grid()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('loss', color='blue')
ax2.set_ylabel('accuracy', color='red')
plt.show()






# config = common.load_config(os.path.join(common.project_root, '../AConvNet_MSTAR_experiments/config/AConvNet-SOC.json'))
# model_name = config['model_name']
# test_set = load_dataset('dataset', False, 'soc', 100)

# m = AConvNet_model.Model(
#     classes=config['num_classes'], channels=config['channels'],
#)

# model_history = glob.glob(os.path.join(common.project_root, f'../AConvNet_MSTAR_experiments/model/{model_name}/*.pth'))
# model_history = sorted(model_history, key=os.path.basename)

# best = {
#     'epoch': 0,
#     'accuracy': 0,
#     'path': ''
# }

# for i, model_path in enumerate(model_history):
#     m.load(model_path)
#     accuracy = evaluate(m, test_set)
#     if accuracy > best['accuracy']:
#         best['epoch'] = i
#         best['accuracy'] = accuracy
#         best['path'] = model_path
#         print(f'Best accuracy at epoch={i} with {accuracy:.2f}%')
    
# best_epoch = best['epoch']
# best_accuracy = best['accuracy']
# best_path = best['path']

# print(f'Final model is epoch={best_epoch} with accurayc={best_accuracy:.2f}%')
# print(f'Path={best_path}')



# best_path = "../AConvNet_MSTAR_experiments/model/AConvNet-SOC/model-038-best.pth"


# m.load(best_path)
# _conf_mat = confusion_matrix(m, test_set)

# sns.reset_defaults()
# ax = sns.heatmap(_conf_mat, annot=True, fmt='d', cbar=False)
# ax.set_yticklabels(mstar.target_name_soc, rotation=0)
# ax.set_xticklabels(mstar.target_name_soc, rotation=30)

# plt.xlabel('prediction', fontsize=12)
# plt.ylabel('label', fontsize=12)


# plt.show()