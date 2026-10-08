import numpy as np
import random
from models import *
from functions import *

np.random.seed(42)
random.seed(42)

## Create training data ##

input_keys=['man_s','dog_s','walks_v','bites_v','dog_o','man_o','_']
input_dict = {
    'man_s':    [1,0,0,0,0],
    'dog_s':    [0,1,0,0,0],
    'walks_v':  [0,0,1,0,0],
    'bites_v':  [0,0,0,1,0],
    'dog_o':    [0,1,0,0,0],
    'man_o':    [1,0,0,0,0],
    '_':        [0,0,0,0,1]
}

grammarmat=np.zeros((7,7))
grammarmat[0,2] = .75; grammarmat[0,3]=.25;
grammarmat[1,2] = .25; grammarmat[1,3]=.75;
grammarmat[2,4] = .75; grammarmat[2,5]=.25;
grammarmat[3,4] = .25; grammarmat[3,5]=.75;
grammarmat[4,6] = 1; 
grammarmat[5,6] = 1; 
grammarmat[6,0] = .5; grammarmat[6,1]=.5;

train_data = []
train_labels = []
input_key = '_'

for _ in range(1000):
  index1 = input_keys.index(input_key)
  input_key = random.choices(input_keys, weights=grammarmat[index1])[0]
  index2 = input_keys.index(input_key) 

  train_labels.append(input_key)
  train_data.append(input_dict[input_key])


## Train model ##

model = FalandaysNetwork(input_nnodes=len(train_data[0]), nnodes=100, p_link=0.1, leak=0.75, lrate_wmat=0.1, lrate_targ=0.01, targ_min=1) 
results = model.run(train_data)


## Cue trained model ##

cue1 = [input_dict["man_s"], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
cue2 = [input_dict["dog_s"], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]

cue1_labels = ["man_s", "NA", "NA", "NA"]
cue2_labels = ["dog_s", "NA", "NA", "NA"]

cue1_results = model.run(cue1, learn_on=False, reset=True)
cue2_results = model.run(cue2, learn_on=False, reset=True)

cue1_cor_matrix = correlate_states_F(results[0], cue1_results[0])
cue2_cor_matrix = correlate_states_F(results[0], cue2_results[0])

plot_correlations_F(cue1_cor_matrix, train_labels, cue1_labels, save=True)
plot_correlations_F(cue2_cor_matrix, train_labels, cue2_labels, save=True)

