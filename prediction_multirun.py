import time
from datetime import timedelta
import numpy as np
import pandas as pd
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from models import *
from functions import *


def run_simulation(sim):
  seed = 42 + sim
  np.random.seed(seed)
  random.seed(seed)

  # generate input sequence
  train_data, train_labels = input_sequence(steps=iterations, start='_c')
  
  # run model
  model = AllostaticNet(input_nnodes=len(train_data[0]), nnodes=100, p_link=0.1, leak=0.75, lrate_wmat=1, lrate_targ=0.01, targ_min=1) 
    # model = AllostaticNetMulti(input_nnodes=len(train_data[0]), subnetworks=2, nnodes=100, p_interlink=coupling, p_link=0.1, leak=0.75, lrate_wmat=0.2, lrate_targ=0.01, targ_min=1)

  results = model.run(train_data, learn_on=True, reset=False)
  
  # store spikes
  results.T.to_csv(f"data/wmat1/log_spikes_{iterations}_sim{sim}.csv", index=False)

  # results = pd.concat([results_pre, results_post], ignore_index=True)

  # for s in range(model.subnetworks):
  #    subnet_df = results.xs(s, axis=1, level="subnetwork").copy()
  #    subnet_df["coupling"] = ([coupling] * midpoint + [0] * (iterations - midpoint))
  #    subnet_df.to_csv(f"data/wmat1/log_spikes_{iterations}_sim{sim}_subnet{s}.csv", index=False)

  pd.DataFrame(train_labels, columns=["label"]).to_csv(f"data/wmat1/log_labels_{iterations}_sim{sim}.csv", index=False)
  
  return sim


## START ##

simulations = 10
iterations = 48000 # length of input sequence
# coupling = 0.025

start_time = time.time()

if __name__ == "__main__": 
    with ProcessPoolExecutor(max_workers=2, max_tasks_per_child=1) as executor:
        futures = [executor.submit(run_simulation, sim) for sim in range(simulations)]

        for future in as_completed(futures):
            sim = future.result()
            print(f"Simulation {sim + 1}/{simulations} completed")

    elapsed = time.time() - start_time
    print(f"Execution time: {str(timedelta(seconds=elapsed))}")