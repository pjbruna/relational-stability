import numpy as np
import pandas as pd
import copy


##################################
##################################


class AllostaticNet:
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
      np.fill_diagonal(self.link_mat, False)  # no self-connections
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
    log_spikes = [] # pd.DataFrame()
    # log_acts = [] # pd.DataFrame()
    # log_wmat = [] # pd.DataFrame()

    # run model on data
    for row in range(len(train_data)):
      # if row % 500 == 0:
      #   print(f'{row}... / {len(train_data)}')

      prev_spikes = self.spikes.copy()
      input = train_data[row]

      errors = self.get_acts(input)

      if learn_on==True:
        self.learning(prev_spikes, errors)

      log_spikes.append(self.spikes.copy())
      # log_acts.append(self.acts.copy())
      # log_wmat.append(self.wmat.flatten())

      # log_spikes = pd.concat([log_spikes, pd.Series(self.spikes)], ignore_index=True, axis=1)
      # log_acts = pd.concat([log_acts, pd.Series(self.acts)], ignore_index=True, axis=1)
      # log_wmat = pd.concat([log_wmat, pd.Series(self.wmat.flatten())], ignore_index=True, axis=1)

    # return state of reservoir
    if reset==True:
        self.spikes = end_spikes
        self.targets = end_targets
        self.acts = end_acts
        self.wmat = end_wmat

    return pd.DataFrame(log_spikes).T # , pd.DataFrame(log_acts).T, pd.DataFrame(log_wmat).T


##################################
##################################


class AllostaticNetMulti:
    def __init__(self, input_nnodes=None, subnetworks=None, nnodes=None, p_interlink=None, p_link=None, leak=None, lrate_wmat=None, lrate_targ=None, targ_min=None):
        self.input_nnodes = input_nnodes
        self.subnetworks = subnetworks
        self.nnodes = nnodes
        self.p_interlink = p_interlink
        self.p_link = p_link
        self.leak = leak
        self.lrate_wmat = lrate_wmat
        self.lrate_targ = lrate_targ
        self.targ_min = targ_min

        S = subnetworks
        N = nnodes

        # creat input weights per subnetwork
        self.input_wmats = [np.random.choice([0.0, 5.0], size=(input_nnodes, N), p=[1 - p_link, p_link]) for _ in range(S)]

        # create internal weights per subnetwork
        self.link_mats = []
        self.wmats = []

        for _ in range(S):
            links = np.random.random((N, N)) < p_link
            np.fill_diagonal(links, False)  # no self-connections
            weights = np.random.normal(0, 1, size=(N, N))
            weights *= links

            self.link_mats.append(links)
            self.wmats.append(weights)

        # create cross-subnetwork weights; crosswmats[i][j] connects subnetwork i to subnetwork j.
        self.crosswmats = [[np.zeros((N, N)) if i == j else (np.random.normal(0, 1, size=(N, N)) * (np.random.random((N, N)) < p_interlink)) for j in range(S)] for i in range(S)]

        # initialize subnetwork states
        self.acts = np.zeros((S, N))
        self.spikes = np.zeros((S, N))
        self.targets = np.full((S, N), targ_min, dtype=float)

    def get_acts(self, input):
        prev_spikes = self.spikes.copy()

        # calculate each subnetwork's new activation
        new_acts = np.zeros_like(self.acts)

        for j in range(self.subnetworks):
            input_drive = input @ self.input_wmats[j]
            internal_drive = prev_spikes[j] @ self.wmats[j]

            cross_drive = np.zeros(self.nnodes)
            for i in range(self.subnetworks):
                if i != j:
                   cross_drive += (prev_spikes[i] @ self.crosswmats[i][j])

            new_acts[j] = (self.leak * self.acts[j] + input_drive + internal_drive + cross_drive)

        self.acts = new_acts

        # log spikes
        thresholds = 2.0 * self.targets
        self.spikes = (self.acts >= thresholds).astype(float)

        # dissipate activity on spiking nodes
        self.acts -= self.spikes * thresholds
        self.acts = np.maximum(self.acts, 0)

        # calculate error
        return self.acts - self.targets

    def learning(self, prev_spikes, errors):
        # per subnetwork...
        for s in range(self.subnetworks):
            # update weights
            active_neighbors = (self.link_mats[s] * prev_spikes[s][:, None]).sum(axis=0)
            dw = (prev_spikes[s][:, None] * errors[s][None, :] * self.lrate_wmat)
            dw *= self.link_mats[s]
            dw = np.divide(dw, active_neighbors[None, :], out=np.zeros_like(dw), where=active_neighbors[None, :] != 0)
            self.wmats[s] -= dw

            # update thresholds
            self.targets[s] += errors[s] * self.lrate_targ
            self.targets[s] = np.maximum(self.targets[s], self.targ_min)

    def remove_cross_connections(self):
        for i in range(self.subnetworks):
            for j in range(self.subnetworks):
                if i != j:
                    self.crosswmats[i][j][:] = 0

    def run(self, train_data, learn_on=True, reset=False):
        saved_state = None
        if reset:
            saved_state = {
                "acts": self.acts.copy(),
                "spikes": self.spikes.copy(),
                "targets": self.targets.copy(),
                "wmats": copy.deepcopy(self.wmats),
                "crosswmats": copy.deepcopy(self.crosswmats),
            }

        log_spikes = []

        for row in train_data:
            prev_spikes = self.spikes.copy()
            errors = self.get_acts(row)

            if learn_on:
                self.learning(prev_spikes, errors)

            log_spikes.append(self.spikes.copy())

        if reset:
            self.acts = saved_state["acts"]
            self.spikes = saved_state["spikes"]
            self.targets = saved_state["targets"]
            self.wmats = saved_state["wmats"]
            self.crosswmats = saved_state["crosswmats"]

        result = np.asarray(log_spikes)

        columns = pd.MultiIndex.from_product([range(self.subnetworks), range(self.nnodes)], names=["subnetwork", "neuron"])
        spike_df = pd.DataFrame(result.reshape(-1, self.subnetworks * self.nnodes), columns=columns)

        return spike_df
