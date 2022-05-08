__author__ = "Lisa van Staden"

import json
import sys

from data_loading.SpeakerDataLoader import SpeakerDataLoader
from models.Encoder import Encoder
from models.Predictor import Predictor
import torch
from torch.nn import functional as F

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import numpy as np


class XVectorTrainer:

    def __init__(self, config, ckpt_path, language="english_full", gender=False):
        self.config_key = "xvector"
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ckpt_path = ckpt_path

        self.language = language
        self.gender = gender

        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None

        self.encoder = None
        self.predictor = None
        self.optimizer = None

        self.epoch = 0

        self.train_data_loader = None
        self.dev_data_loader = None
        self.test_data_loader = None

    def build(self):
        num_frames = self.config[self.config_key]["num_frames"]
        context_sizes = self.config[self.config_key]["context_sizes"]
        dilations = self.config[self.config_key]["dilations"]
        input_dim = self.config[self.config_key]["input_dim"]
        output_dim = self.config[self.config_key]["output_dim"]
        hidden_dim = self.config[self.config_key]["hidden_dim"]
        segment_layer_dim = self.config[self.config_key]["segment_layer_dim"]
        num_speakers = self.config[self.config_key]["num_speakers"]
        learning_rate = self.config[self.config_key]["learning_rate"]

        self.encoder = Encoder(num_frames=num_frames, context_sizes=context_sizes, dilations=dilations,
                               input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
                               segment_layer_dim=segment_layer_dim).to(
            self.device)

        self.predictor = Predictor(input_dim=hidden_dim, num_speakers=num_speakers).to(self.device)

        model_params = list(self.encoder.parameters()) + list(self.predictor.parameters())

        self.optimizer = torch.optim.Adam(model_params, lr=learning_rate)

    def train(self, num_epochs=-1):
        batch_size = self.config[self.config_key]["batch_size"]
        num_buckets = self.config[self.config_key]["num_buckets"]
        num_epochs = self.config[self.config_key]["num_epochs"] if num_epochs == -1 else num_epochs
        input_dim = self.config[self.config_key]["input_dim"]

        starting_epoch = self.epoch

        if self.train_data_loader is None:
            self.train_data_loader = SpeakerDataLoader(dataset_type="training",
                                                       language=self.language,
                                                       batch_size=batch_size,
                                                       num_buckets=num_buckets, max_seq_len=100,
                                                       include_speaker_ids=not self.gender,
                                                       include_gender_ids=self.gender,
                                                       dframe=input_dim)

        for epoch in range(starting_epoch, num_epochs):
            self.encoder.train()
            self.predictor.train()
            losses = []
            accs = []
            for batch in self.train_data_loader:
                self.optimizer.zero_grad()
                x = batch.X.to(self.device)
                speaker_idxs = batch.speaker_X_idx.type(torch.LongTensor).to(self.device) \
                    if not self.gender else batch.gender_X_idx.type(torch.LongTensor).to(self.device)
                xvector = self.encoder(x)
                probs = self.predictor(xvector)

                loss = F.cross_entropy(probs, speaker_idxs)
                losses.append(loss)
                _, predicted = torch.max(probs, dim=1)
                accs.append(torch.tensor((predicted == speaker_idxs).sum().item() / batch_size))
                loss.backward()

                self.optimizer.step()

            if epoch % 1 == 0:
                print(f'Epoch: {epoch}')
                print(f'Train Loss: {torch.stack(losses).mean()}')
                print(f'Train Accuracy: {torch.stack(accs).mean()}')

            if (epoch + 1) % 100 or (epoch + 1) % num_epochs == 0:
                self.epoch = epoch
                self.save_checkpoint(at_epoch=True)

        # self.save_checkpoint()

    def evaluate(self, verbose=True, data="validation"):

        batch_size = self.config[self.config_key]["batch_size"]
        input_dim = self.config[self.config_key]["input_dim"]

        if data == "validation":
            if self.dev_data_loader is None:
                self.dev_data_loader = SpeakerDataLoader("validation", language=self.language, include_speaker_ids=True,
                                                         batch_size=batch_size, max_seq_len=100, dframe=input_dim)
            data_loader = self.dev_data_loader

        elif data == "test":
            if self.test_data_loader is None:
                self.test_data_loader = SpeakerDataLoader("test", language=self.language, include_speaker_ids=True,
                                                          batch_size=batch_size, max_seq_len=100, dframe=input_dim)
            data_loader = self.test_data_loader

        else:
            print("Wrong data type give")
            sys.exit()

        self.encoder.eval()
        self.predictor.eval()
        losses = []
        accs = []
        for batch in data_loader:
            x = batch.X.to(self.device)
            speaker_idxs = batch.speaker_X_idx.type(torch.LongTensor).to(self.device)
            xvector = self.encoder(x)
            probs = self.predictor(xvector)

            losses.append(F.cross_entropy(probs, speaker_idxs))
            _, predicted = torch.max(probs, dim=1)
            accs.append((predicted == speaker_idxs).sum().item() / batch_size)

        acc = torch.stack(accs).mean()
        loss = torch.stack(losses).mean()

        if verbose:
            print("Validation Accuracy:", acc)
            print("Validation Loss:", loss)

        return acc, loss

    def save_features(self, path, language, dataset="training", projection="lda", draw=True):
        self.encoder.eval()
        dataloader = SpeakerDataLoader(dataset_type=dataset,
                                       language=language,
                                       batch_size=1,
                                       max_seq_len=100,
                                       dframe=39,
                                       include_gender_ids=self.gender)
        speaker_utt_dict = {}

        with torch.no_grad():
            for dp in dataloader:
                x = dp.X.to(self.device)
                speaker_id = dp.speaker_X[0] if not self.gender else ('f' if dp.gender_X_idx == 0 else 'm')
                if dp.X_lengths[0] < 30:
                    continue
                xvector = self.encoder(x)[0].detach().cpu().numpy()

                if speaker_id not in speaker_utt_dict:
                    speaker_utt_dict[speaker_id] = []
                speaker_utt_dict[speaker_id].append(xvector)

            all_utts = []
            speakers = []
            speaker_idxs = []

            for (k, speaker) in enumerate(speaker_utt_dict):
                speakers.append(speaker)
                all_utts += speaker_utt_dict[speaker]
                speaker_idxs += [k for _ in range(len(speaker_utt_dict[speaker]))]

        if projection == "lda":
            lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
            projected_xvecs = lda.fit_transform(all_utts, speaker_idxs)

        if projection == "pca":
            pca = PCA(n_components=10)
            projected_xvecs = pca.fit_transform(all_utts)

        if projection == "mean":
            projected_xvecs = all_utts

        projected_xvecs = np.array(projected_xvecs)
        speaker_idxs = np.array(speaker_idxs)

        xvec_dict = {}
        for k in range(len(speakers)):
            xvec_dict[speakers[k]] = np.mean(projected_xvecs[speaker_idxs == k], axis=0)
            print(xvec_dict[speakers[k]])

        np.savez(path, **xvec_dict)

    def save_checkpoint(self, at_epoch=False):
        path = self.ckpt_path
        if at_epoch:
            path = path[:-5] + f"_{self.epoch}.ckpt"
        torch.save({'epoch': self.epoch + 1,
                    'enc_model_state_dict': self.encoder.state_dict(),
                    'pred_model_state_dict': self.predictor.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                    }, path)

    def load_checkpoint(self, epoch):
        path = self.ckpt_path
        path = path[:-5] + f"_{epoch}.ckpt"
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['enc_model_state_dict'])
