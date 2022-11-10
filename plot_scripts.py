import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.lines import Line2D



def plot_train(data):
    fig_loss = plt.gcf()
    fig_loss.set_size_inches(108.5, 10.5, forward=True)

    plt.rcParams.update({'font.size': 15})
    #plt.plot(data)
    data.plot.line(x='Step', y='MsPacman-v4__my_baseline_intrinsic__1__1664396744 - charts/mean_episodic_return')
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    #plt.grid(True)
    #plt.legend([Line2D([0], [0], color="b", lw=4)], ['Loss Train'])
    #plt.xlim((0, len(data)))
    #plt.ylim((np.ndarray.min(data) - 1, np.ndarray.max(data) + 1))
    plt.show()
    plt.draw()
    #fig_loss.savefig(os.path.join(path, "visuals", "loss_train.pdf"), bbox_inches='tight', pad_inches=0)



''' 
fig_m = plt.gcf()
fig_m.set_size_inches(8., 8.)

plt.rcParams.update({'font.size': 12})
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
#ax.set_title(f'Confusion Matrix {len}')
#ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
#ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
ax.xaxis.set_ticklabels(['0', '1', '2'])
ax.yaxis.set_ticklabels(['0', '1', '2'])
plt.show()
plt.draw()
if not os.path.exists(os.path.join(path, "visuals")):
    os.makedirs(os.path.join(path, "visuals"))
fig_m.savefig(
    os.path.join(path, "visuals", f"confusion_matrix_{split}_{len}.pdf"), bbox_inches='tight', pad_inches=0)
'''

if __name__ == "__main__":
    data = pd.read_csv('results/pacman_my_intrinsic.csv')

    print(data.head())
    print(data.columns)
    plot_train(data)