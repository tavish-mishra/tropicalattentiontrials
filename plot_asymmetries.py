import pandas as pd
import matplotlib.pyplot as plt

tropical_er = '15_exp/train/train_FloydWarshallDataset_tropical_0.0001_20000_20260502_213237_relu.csv'
tropical_mix = '15_exp/train/train_FloydWarshallDataset_tropical_0.0001_20000_20260503_131700_relu.csv'
vanilla_er = '15_exp/train/train_FloydWarshallDataset_vanilla_0.0001_20000_20260502_215200_relu.csv'
vanilla_mix = '15_exp/train/train_FloydWarshallDataset_vanilla_0.0001_20000_20260503_131700_relu.csv'

files = [tropical_er, tropical_mix, vanilla_er, vanilla_mix]
names = ['Tropical on ER', 'Tropical on Mix', 'Vanilla on ER', 'Vanilla on Mix']
colors = ['#2196F3', '#4CAF50', '#F44336', '#FF9800']

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Attention Asymmetry Over Training', fontsize=20, fontweight='bold', y=1.01)

for ax, file, name, color in zip(axes.flat, files, names, colors):
    df = pd.read_csv(file)
    ax.plot(df['epoch'], df['asymmetry'], color=color, linewidth=2)
    ax.set_title(name, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Asymmetry', fontsize=12)
    ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('asymmetries.png', dpi=150, bbox_inches='tight')
plt.show()
