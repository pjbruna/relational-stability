import numpy as np
import random
import scipy.stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from datetime import datetime

## Generate input sequence for Falandays et al. 2021-style language task ##

def input_sequence(steps=None, start='_c'):
  input_keys=['man_s','dog_s','walks_v','bites_v','dog_o','man_o','_a', '_b', '_c']
  input_dict = {
      'man_s':    [1,0,0,0,0,0,0],
      'dog_s':    [0,1,0,0,0,0,0],
      'walks_v':  [0,0,1,0,0,0,0],
      'bites_v':  [0,0,0,1,0,0,0],
      'dog_o':    [0,1,0,0,0,0,0],
      'man_o':    [1,0,0,0,0,0,0],
      '_a':       [0,0,0,0,1,0,0],
      '_b':       [0,0,0,0,0,1,0],
      '_c':       [0,0,0,0,0,0,1]
  }

  grammarmat=np.zeros((9,9))
  grammarmat[0,2] = .75; grammarmat[0,3]=.25;
  grammarmat[1,2] = .25; grammarmat[1,3]=.75;
  grammarmat[2,4] = .75; grammarmat[2,5]=.25;
  grammarmat[3,4] = .25; grammarmat[3,5]=.75;
  grammarmat[4,6] = 1; 
  grammarmat[5,6] = 1; 
  grammarmat[6,7] = 1; 
  grammarmat[7,8] = 1; 
  grammarmat[8,0] = .5; grammarmat[8,1]=.5;

  data = []
  labels = []
  input_key = start

  for _ in range(steps):
    index1 = input_keys.index(input_key)
    input_key = random.choices(input_keys, weights=grammarmat[index1])[0]
    index2 = input_keys.index(input_key) 

    data.append(input_dict[input_key])
    labels.append(input_key)

  return data, labels


## Falandays et al. 2021 analyses ##

def correlate_states_F(train_data, cue_data):
  plot_data = np.array(train_data.iloc[:,-12:])
  plot_data = np.hstack((plot_data, np.array(cue_data)))
  cor_matrix = np.zeros((16,16))
  for n in range(16):
    x = plot_data[:,n]
    for r in range(16):
      if (r >= n):  
        y = plot_data[:,r]
        res = np.around(scipy.stats.pearsonr(x, y)[0],2)
        cor_matrix[r,n]=res
      else:
        cor_matrix[r,n] = np.nan

  return cor_matrix


def plot_correlations_F(r_mat, train_labels, cue_labels, save=True):
  cmap = cm.get_cmap('RdYlBu', 7)

  fig, ax = plt.subplots()
  im = ax.imshow(r_mat, cmap=cmap)

  ax.set_xticks(np.arange(16))
  ax.set_yticks(np.arange(16))

  ax.set_xticklabels(np.append(np.array(train_labels[-12:]), np.array(cue_labels)).tolist())
  ax.set_yticklabels(np.append(np.array(train_labels[-12:]), np.array(cue_labels)).tolist())

  plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

  for i in range(16):
    for j in range(16):
      if j > i:
        continue
      text = ax.text(j, i, r_mat[i, j], ha="center", va="center", color="black")

  plt.colorbar(im, ax=ax)
  im.set_clim(-1, 1)
  fig.tight_layout()
  fig.set_size_inches(13.75, 10)

  if save==True:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    plt.savefig(f'figures/F_cormat_{timestamp}.png')
  else:
    plt.show()


## PCA ##

def plot_PCA(train_data, train_labels, steps=100, start=None, save=True):
    if start is None:   # use last {steps} timesteps if {start} not specified
      start = train_data.shape[1] - steps
    end = start + steps
    
    plot_data = np.array(train_data.iloc[:, start:end]).T
    labels = np.array(train_labels[start:end]).ravel()

    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(plot_data)
    variance = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(
            pca_data[mask, 0],
            pca_data[mask, 1],
            pca_data[mask, 2],
            label=label,
            alpha=0.8
        )

    ax.set_xlabel(f'PC1 ({variance[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({variance[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({variance[2]*100:.1f}%)')
    ax.legend()

    plt.tight_layout()

    if save==True:
      timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
      plt.savefig(f'figures/PCA_{start}-{end}steps_{timestamp}.png')
    else:
      plt.show()

    # return pca_data, pca.explained_variance_ratio_
