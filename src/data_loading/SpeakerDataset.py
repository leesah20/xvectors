__author__ = "Lisa van Staden"

import torch
from torch.utils import data


class SpeakerDataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, npz, language="english_full", d_frame=39, max_seq_len=100, speaker=False, gender=False,
                 speaker_gender_dict: dict = None):
        """

        :type speaker_gender_dict: object
        """

        print("Creating Speaker Dataset")

        self.language = language
        self.speaker = speaker
        self.gender = gender
        self.speaker_gender_dict = speaker_gender_dict
        self.d_frame = d_frame
        self.max_seq_len = max_seq_len
        self.num_speakers = 0
        self.num_genders = 0
        self.utt_keys = list(set([key.split(".")[0] for key in sorted(npz)]))
        self.labels = [utt_key.split("_")[0] for utt_key in self.utt_keys]
        if speaker:
            self.speaker_to_id_map = {}
            self.id_to_speaker_map = {}
            self._set_num_speakers()
            print("Num speakers: ", self.num_speakers)
        if gender:
            self.num_genders = len(list(set(self.speaker_gender_dict.values())))
            print("num genders: ", self.num_genders)
        self.data = {}
        for key in self.utt_keys:
            if "_cpc_feats" in self.language or "capc_feats" in self.language:
                self.data[key] = torch.as_tensor(npz[f"{key}.c"][:self.max_seq_len, :])
            elif "aligned" in self.language:
                self.data[f"{key}.X"] = torch.as_tensor(npz[f"{key}.X"][:self.max_seq_len, :d_frame],
                                                        dtype=torch.float32)
                self.data[f"{key}.Y"] = torch.as_tensor(npz[f"{key}.Y"][:self.max_seq_len, :d_frame],
                                                        dtype=torch.float32)
            else:
                self.data[key] = torch.as_tensor(npz[key][:self.max_seq_len, :d_frame], dtype=torch.float32)

    def get_num_speakers(self):
        print("getting number of speakers from dataset")
        return self.num_speakers

    def _set_num_speakers(self):
        for key in self.utt_keys:
            if self.language == "english_full":
                speaker = key.split("_")[0][:3]
            elif self.language.startswith("english"):
                speaker = key.split("_")[1][:3]
            elif self.language == "xitsonga_cpc":
                speaker = key.split("-")[2]
            elif self.language.startswith("xitsonga"):
                speaker = key.split("_")[1].split("-")[2]
            elif self.language == "hausa":
                speaker = key.split("_")[1].lower()
            self.speaker_to_id(speaker)

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, index):
        sample = {"index": index}
        utt_key1 = self.utt_keys[index]
        sample["utt_key"] = utt_key1
        sample["X"] = self.data[utt_key1]
        sample["X_length"] = len(sample["X"])
        sample["word"] = utt_key1.split("_")[0]

        if self.language == "english":
            sample["speaker_X"] = utt_key1.split("_")[1][:3]
        elif self.language.startswith("xitsonga"):
            sample["speaker_X"] = utt_key1.split("_")[1].split("-")[2]
        elif self.language == "hausa":
            sample["speaker_X"] = utt_key1.split("_")[1].lower()
        elif self.language == "english_full":
            sample["speaker_X"] = utt_key1.split("_")[0][:3]

        if self.speaker:
            sample["speaker_X_idx"] = self.speaker_to_id(sample["speaker_X"])

        if self.language.startswith("english") or self.language == "hausa":
            if self.gender:
                sample["gender_X_idx"] = 0 if self.speaker_gender_dict[sample["speaker_X"]] == 'f' else 1

        elif self.language == "xitsonga":
            if self.gender:
                sample["gender_X_idx"] = 0 if sample["speaker_X"][-1] == 'f' else 1

        return sample
