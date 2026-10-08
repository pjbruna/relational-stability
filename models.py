import numpy as np
import pandas as pd

class FalandaysNetwork:
  def __init__(self, input_nnodes=None, nnodes=None, p_link=None, leak=None, lrate_wmat=None, lrate_targ=None, targ_min=None):
    # model hyperparameters
    self.nnodes = nnodes
    self.p_link = p_link
    self.leak = leak
    self.lrate_wmat = lrate_wmat
    self.lrate_targ = lrate_targ
    self.targ_min = targ_min

    if input_nnodes is not None:
      # creat input layer and weights
      self.input_wmat = np.random.choice([0,5], size=(input_nnodes, nnodes), p=[1-p_link, p_link])

      # create internal weight matrix
      self.link_mat = np.random.choice([0,1], size=(nnodes, nnodes), p=[1-p_link, p_link])
      self.wmat = np.where(self.link_mat == 1, np.random.normal(0, 1, size=(nnodes, nnodes)), 0)

      # initialize reservoir state
      self.acts = np.zeros(nnodes)
      self.spikes = np.zeros(nnodes)
      self.targets = np.repeat(targ_min, nnodes)
  

  def get_acts(self, input):
    # update activation in reservoir
    self.acts = self.acts*self.leak + np.dot(input, self.input_wmat) + np.dot(self.spikes, self.wmat)

    # log spikes
    thresholds = self.targets*2
    self.spikes[self.acts>=thresholds] = 1
    self.spikes[self.acts<thresholds] = 0

    # dissipate activation on spiking nodes
    self.acts[self.spikes==1] = self.acts[self.spikes==1] - thresholds[self.spikes==1]
    self.acts[self.acts<0] = 0

    # log errors
    errors = self.acts-self.targets

    return errors


  def learning(self, prev_spikes, errors):
    # track which neighbors spiked
    prev_inactive = np.argwhere(prev_spikes<=0)[:,0]

    # apply learning rules
    active_neighbors = self.link_mat.copy()
    active_neighbors[prev_inactive,:] = 0
    active_neighbors = np.sum(active_neighbors, axis=0)

    d_wmat = np.zeros((self.nnodes, self.nnodes))
    d_wmat[:,:] = errors*self.lrate_wmat
    d_wmat[self.link_mat==0] = 0
    d_wmat[prev_inactive,:] = 0
    d_wmat = np.divide(d_wmat, active_neighbors, out=np.zeros_like(d_wmat, dtype=np.float64), where=active_neighbors != 0)

    self.wmat -= d_wmat

    self.targets = self.targets + (errors*self.lrate_targ)
    self.targets[self.targets<self.targ_min] = self.targ_min


  def run(self, train_data, learn_on=True, reset=False):
    # log current state of reservoir
    end_spikes = self.spikes.copy()
    end_targets = self.targets.copy()
    end_acts = self.acts.copy()
    end_wmat = self.wmat.copy()

    # store data
    log_spikes = pd.DataFrame()
    log_acts = pd.DataFrame()
    log_wmat = pd.DataFrame()

    # run model on data
    for row in range(len(train_data)):
      prev_spikes = self.spikes.copy()
      input = train_data[row]

      errors = self.get_acts(input)

      if learn_on==True:
        self.learning(prev_spikes, errors)

      log_spikes = pd.concat([log_spikes, pd.Series(self.spikes)], ignore_index=True, axis=1)
      log_acts = pd.concat([log_acts, pd.Series(self.acts)], ignore_index=True, axis=1)
      log_wmat = pd.concat([log_wmat, pd.Series(self.wmat.flatten())], ignore_index=True, axis=1)

    # return state of reservoir
    if reset==True:
        self.spikes = end_spikes
        self.targets = end_targets
        self.acts = end_acts
        self.wmat = end_wmat

    return log_spikes, log_acts, log_wmat
  