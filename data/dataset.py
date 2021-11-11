import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MedicalPretrainingDataset(Dataset):

    def __init__(self, input_file):
        self.input_file = input_file
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'seq_level_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, seq_level_labels] = [
            torch.from_numpy(inputs[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(inputs[index].astype(np.int64))) for indice, inputs in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        # store number of masked tokens in index
        padded_mask_indices = masked_lm_positions.eq(0).nonzero()
        if len(padded_mask_indices) > 0:
            index = padded_mask_indices[0].item()
        else:
            index = len(masked_lm_positions)
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, seq_level_labels]


