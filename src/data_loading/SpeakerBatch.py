from data_loading import helpers

__author__ = "Lisa van Staden"

import torch
from torch import nn


class SpeakerBatch:
    def __init__(self, data):

        self.data = data

        self.labels = [d["word"] for d in self.data]

        self.utt_key = [d["utt_key"] for d in self.data]

        self.X_lengths = torch.stack(self._get_data("X_length"))

        self.max_length = torch.max(self.X_lengths).item()

        self.X = self._assign("X")

        self.speaker_X = [d["speaker_X"] for d in self.data]

        self.indices = [d["index"] for d in self.data]

        if "speaker_X_idx" in data[0]:
            self.speaker_X_idx = torch.stack(self._get_data("speaker_X_idx"))

        if "gender_X_idx" in data[0]:
            self.gender_X_idx = torch.stack(self._get_data("gender_X_idx"))

        if "Y" in data[0]:

            self.Y_lengths = torch.stack(self._get_data("Y_length"))

            max_y_length = torch.max(self.Y_lengths).item()

            self.max_length = max(max_y_length, self.max_length)

            self.speaker_Y = [d["speaker_Y"] for d in self.data]

            self.Y = self._assign("Y")

            if "speaker_Y_idx" in data[0]:
                self.speaker_Y_idx = torch.stack(self._get_data("speaker_Y_idx"))

            if "gender_Y_idx" in data[0]:
                self.gender_Y_idx = torch.stack(self._get_data("gender_Y_idx"))

    def _get_data(self, key):
        return [torch.as_tensor(d[key], dtype=helpers.DTYPE["TORCH_FLOAT"]) for d in self.data]

    def _assign(self, key):
        return nn.utils.rnn.pad_sequence(self._get_data(key), batch_first=True)
