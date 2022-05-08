import os
import random
import glob

__author__ = "Lisa van Staden"

from torch.utils import data
import torch
import json
import numpy as np
import sys

from data_loading.BucketSampler import BucketSampler
from data_loading.SpeakerDataset import SpeakerDataset
from data_loading.SpeakerBatch import SpeakerBatch
from data_loading.SpeakerBucketSampler import SpeakerBucketSampler


def _get_default_speaker_to_gender_map():
    speaker_gender_dict = {
        "s01": "f", "s02": "f", "s03": "m", "s04": "f", "s05": "f", "s06": "m",
        "s07": "f", "s08": "f", "s09": "f", "s10": "m", "s11": "m", "s12": "f",
        "s13": "m", "s14": "f", "s15": "m", "s16": "f", "s17": "f", "s18": "f",
        "s19": "m", "s20": "f", "s21": "f", "s22": "m", "s23": "m", "s24": "m",
        "s25": "f", "s26": "f", "s27": "f", "s28": "m", "s29": "m", "s30": "m",
        "s31": "f", "s32": "m", "s33": "m", "s34": "m", "s35": "m", "s36": "m",
        "s37": "f", "s38": "m", "s39": "f", "s40": "m"
    }

    return speaker_gender_dict


def _parse_spk_file(spk_file):
    info_dict = {}
    with open(spk_file, encoding='latin_1') as spk:
        pairs = spk.read().split('\n')
        for pair in pairs:
            if len(pair.split(':')) == 2:
                [key, value] = pair.split(':')
                info_dict[key[1:].lower()] = value.lower()
    return info_dict


def _get_spk_to_info_map(spk_dir):
    speaker_info_dict = {}
    for spk_file in glob.glob(f'{spk_dir}/*.spk'):
        speaker_info_dict[spk_file.split('/')[-1].split('.')[0].lower()] = _parse_spk_file(spk_file)
    return speaker_info_dict


class SpeakerDataLoader(data.DataLoader):
    dataset: SpeakerDataset

    def __init__(self, dataset_type, batch_size=1, include_speaker_ids=False, include_gender_ids=False,
                 speaker_to_gender_map=None, num_buckets=3, language="english_full",
                 max_seq_len=100, dframe=39,
                 ):
        """
        :param dataset_type: validation or training
        :param batch_size: Set batch_size=0 for no mini-batching
        :param pairs:
        :param include_speaker_ids:
        :param include_gender_ids:
        :param speaker_to_gender_map:
        """

        if os.getcwd() == "/home":
            root = "/home"
        else:
            root = "../.."
        npz = None
        with open(root + "/config/data_paths.json") as paths_file:
            path_dict = json.load(paths_file)
            spk_dir = f'{root}{path_dict.get(f"{language}_spk_files")}'
            if dataset_type == "training":
                npz = np.load(path_dict["{0}_train_data".format(language)])
            elif dataset_type == "validation" and not language.startswith("xitsonga"):
                npz = np.load(path_dict["{0}_validation_data".format(language)])
            elif dataset_type == "test":
                npz = np.load(path_dict["{0}_test_data".format(language)])

            else:
                sys.exit("Invalid dataset type given.")

        if npz is not None:

            if language == 'hausa':
                speaker_to_gender_map = {k: ('f' if v['sex'] == 'female' else 'm')
                                         for (k, v) in _get_spk_to_info_map(spk_dir).items()}

            self.dataset = SpeakerDataset(npz, language=language, speaker=include_speaker_ids,
                                          gender=include_gender_ids,
                                          speaker_gender_dict=speaker_to_gender_map
                                                              or _get_default_speaker_to_gender_map(),
                                          max_seq_len=max_seq_len, d_frame=dframe)

            if batch_size == 0:
                batch_size = len(self.dataset)

            if (dataset_type != "training") and language != 'english_full':
                super(SpeakerDataLoader, self).__init__(self.dataset, shuffle=False,
                                                        collate_fn=(lambda sp: SpeakerBatch(sp)),
                                                        batch_size=batch_size, drop_last=True)
            else:
                self.sampler = BucketSampler(self.dataset, num_buckets)
                super(SpeakerDataLoader, self).__init__(self.dataset,
                                                        sampler=self.sampler,
                                                        collate_fn=(lambda sp: SpeakerBatch(sp)), batch_size=batch_size,
                                                        drop_last=True)

    def get_num_speakers(self):
        return self.dataset.get_num_speakers()
