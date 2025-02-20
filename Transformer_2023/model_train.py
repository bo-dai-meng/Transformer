import pandas as pd
from torch import nn
import torch
import matplotlib.pyplot as plt
from torch.utils import data
from torch.utils.data import DataLoader
import dataset
import Transformer_model
import plot_pred

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight)

def train(args):
    emodel = Transformer_model.Transformer(args).to(
        device=torch.device('cuda'))
    emodel.apply(init_weights)
    loss = nn.MSELoss().to(device=torch.device('cuda'))
    optimizer = torch.optim.Adam(emodel.parameters(), args.lr, weight_decay=args.weight_decay)
    label_list = []

    dataloader = dataset.Read_Data()
    dataloader = dataloader.get_dataloader(args.batch_sizes)
    emodel.train()
    train_loss_list = []
    for i in range(args.epoch):
        # count = 0
        for datas in dataloader:
            # count = count + 1
            optimizer.zero_grad()
            feature, label = datas
            feature = feature.float().cuda()
            label = label.float().cuda()
            print(f"\nfeature_shape：{feature.shape}, label_shape：{label.shape}")
            output = emodel(feature[:, :, 0:4], feature[:, :, 4].reshape(args.batch_sizes, args.time_step, -1), None)
            print(f"\nprediction:{output}，output_shape：{output.shape}")
            train_loss = loss(output, label.reshape(args.batch_sizes, -1))
            print(f"\nloss:{train_loss}")
            train_loss_list.append(train_loss.item())
            train_loss.backward()
            optimizer.step()

    test_dataloader = dataset.Read_Data(filepath = 'D:\\Deeplearning\\dl\\Transformer-project02\\Transformer_2023\\Al_ion_battery_data\\1-test.xlsx', sheet_name='Sheet1', time_step= 10)
    test_dataloader = test_dataloader.get_dataloader(1)
    emodel.eval()
    with torch.no_grad():
        output_list = []
        for datas in test_dataloader:
            feature, label = datas
            print(feature.shape, label.shape)
            label_list.append(label.item())
            feature = feature.float().cuda()
            label = label.float().cuda()
            output = emodel(feature[:, :, 0:4], feature[:, :, 4].reshape(1, args.time_step, -1), None)
            output = output.cpu()
            print(output, output.shape)
            output_list.append(output.item())

    plot_pred.plot(12, 10, train_loss_list, output_list, label_list)


