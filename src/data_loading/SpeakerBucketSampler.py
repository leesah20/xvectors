__author__ = "Lisa van Staden"

import random

from torch.utils import data

from data_loading.SpeakerDataset import SpeakerDataset


class SpeakerBucketSampler(data.Sampler):

    def __init__(self, data_source: SpeakerDataset, num_speakers, num_utts, num_buckets=3, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_buckets = num_buckets
        self.num_utts = num_utts
        self.buckets = []
        self.shuffle = shuffle
        self._allocate_buckets()

    def __iter__(self):
        # print(self.shuffle)
        if not self.shuffle:
            random.seed(0)
        buckets = sum([self.buckets[i].get_utterances() for i in range(self.num_buckets)], [])
        # print(buckets[0])
        return iter(buckets)

    def __len__(self):
        return len(self.data_source)

    def _allocate_buckets(self):
        buckets = [[] for _ in range(self.num_buckets)]
        seq_lengths = [(i, self.data_source[i].get("X_length")) for i in range(len(self.data_source))]
        lengths = list(seq_len for (i, seq_len) in seq_lengths)
        lengths.sort()
        bucket_means = []
        if len(lengths) < self.num_buckets:
            print("Error: Too many buckets")
        for i in range(self.num_buckets):
            bucket_means.append(lengths[(i + 1) * int(len(lengths) / (self.num_buckets + 1))])

        for (i, seq_len) in seq_lengths:
            dists = [abs(seq_len - bm) for bm in bucket_means]
            index = dists.index(min(dists))
            buckets[index].append(i)

        for i in range(self.num_buckets):
            self.buckets.append(SpeakerBucket(buckets[i], self.data_source, self.num_utts))


class SpeakerBucket:
    def __init__(self, utt_idxs, data_source, num_utts_per_speaker):
        self.utt_idxs = utt_idxs
        self.data_source = data_source
        self.speaker_utts = {}
        self.num_utts = num_utts_per_speaker
        self.utts_count = []
        self._allocate_speakers()

    def _allocate_speakers(self):
        for utt_idx in self.utt_idxs:
            speaker_idx = self.data_source[utt_idx].get("speaker_X_idx")
            if speaker_idx not in self.speaker_utts:
                self.speaker_utts[speaker_idx] = []
            self.speaker_utts[speaker_idx].append(utt_idx)
        pop_list = []
        for k in self.speaker_utts:
            self.utts_count.append(len(self.speaker_utts[k]))
            if len(self.speaker_utts[k]) < self.num_utts:
                pop_list.append(k)
        for pop_idx in pop_list:
            self.speaker_utts.pop(pop_idx)

    def get_utterances(self):
        batch_speakers = list(self.speaker_utts.keys())
        random.shuffle(batch_speakers)
        return sum([random.sample(self.speaker_utts[i], self.num_utts) for i in batch_speakers], [])

