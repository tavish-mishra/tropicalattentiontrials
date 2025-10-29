import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

tropical_loss_3_layer = pd.read_csv('15_exp/train/train_FloydWarshallDataset_tropical_0.0001_100_20251028_150142_relu.csv').loc[:, 'loss']
tropical_loss_1_layer = pd.read_csv('15_exp/train/train_FloydWarshallDataset_tropical_0.0001_100_20251027_105301_relu.csv').loc[:, 'loss']

vanilla_loss_3_layer = pd.read_csv('15_exp/train/train_FloydWarshallDataset_vanilla_0.0001_100_20251028_150131_relu.csv').loc[:, 'loss']
vanilla_loss_1_layer = pd.read_csv('15_exp/train/train_FloydWarshallDataset_vanilla_0.0001_100_20251027_105302_relu.csv').loc[:, 'loss']


name_dict = {
    'Tropical Loss (3 Layer)': tropical_loss_3_layer,
    'Vanilla Loss (3 Layer)': vanilla_loss_3_layer,
    'Vanilla Loss (1 Layer)': vanilla_loss_1_layer,
    'Tropical Loss (1 Layer)': tropical_loss_1_layer,
}

for name in name_dict.keys():
    loss = name_dict[name]
    title = f'{name} MSE'
    x_name = 'Global Step (Cumulative Batch Index)'
    y_name = 'MSE Loss'
    x_axis = np.arange(len(loss))
    plt.figure()  # Create a new figure for each plot
    plt.plot(x_axis, loss, label=name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(f'15_exp/plots/{title.replace(" ", "_")}.png')