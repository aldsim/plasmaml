"""
PEALDProfile Pytorch Dataset
"""

import numpy as np

import torch
from torch.utils.data import Dataset

import os

class PEALDProfile(Dataset):

    def __init__(self, profile_file, tsat_file, beta_file, betarec_file,
                 filter_p=0):

        self.profile = np.load(profile_file)
        self.tsat = np.load(tsat_file)
        self.tsat = self.tsat.reshape((len(self.tsat),1))
        self.filter = filter_p
        if self.filter > 0:
            self.indices = np.arange(0, 20, self.filter)
            p1 = self.profile[:,:20]
            p2 = self.profile[:,20:]
            p1 = p1[:,self.indices]
            p2 = p2[:,self.indices]
            self.profile = np.concatenate([p1,p2], axis=1)

        self.beta = np.log(np.load(beta_file))
        self.beta = self.beta.reshape((len(self.beta),1))

        self.betarec = np.log(np.load(betarec_file))
        self.betarec = self.betarec.reshape((len(self.betarec),1))

    def __len__(self):
        return self.profile.shape[0]

    def __getitem__(self, idx):
        profs = torch.tensor(self.profile[idx,:],dtype=torch.float)
        tsat = torch.tensor(self.tsat[idx], dtype=torch.float)
        beta = torch.tensor(self.beta[idx], dtype=torch.float)
        betarec = torch.tensor(self.betarec[idx], dtype=torch.float)

        return profs, tsat, beta, betarec


class PEALD(PEALDProfile):

    def __init__(self, train=True, directory="./dataset", filter_p=0):
        if train:
            profile_file = os.path.join(directory, "train_data.npy")
            tsat_file = os.path.join(directory, "train_labels.npy")
            beta_file = os.path.join(directory, "train_beta.npy")
            betarec_file = os.path.join(directory, "train_betarec.npy")

        else:
            profile_file = os.path.join(directory, "test_data.npy")
            tsat_file = os.path.join(directory, "test_labels.npy")
            beta_file = os.path.join(directory, "test_beta.npy")
            betarec_file = os.path.join(directory, "test_betarec.npy")

        super().__init__(profile_file, tsat_file, beta_file, betarec_file,
                         filter_p=filter_p)

