from typing import Tuple, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from classification.supports import utils
from classification.supports.monitor import Monitor
from classification.functions.train_base import BaseTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_vqvae = torch.load('results/finetune_model_pth/finetune-clf-1/vqvae_final.pth', map_location = device)
# decoder_new = torch.load('results/finetune_model_pth/finetune-clf-1/decoder_new_final.pth', map_location = device)
# model_vqvae.eval()
# decoder_new.eval()

def model_vqave_preprocess(data, model_vqvae, decoder_new):
    with torch.no_grad():
            # input data: (batch, 5000, 12)
            z = model_vqvae._encoder(data) # z keys: 'last_hidden_state', 'loc', 'scale', 'patch_input'

            # z.last_hidden_state: (batch, 12, patches, patch_length)
            hidden_state = model_vqvae.BN(z.last_hidden_state)

            quantization, indices, loss = model_vqvae._vq(hidden_state) # (batch, 12, 625/500, 128)

            quantized_de = model_vqvae._map(quantization).reshape(data.shape[0], 12, -1) # batch, 12, 5000

            if quantized_de.shape[-1] < 5000:
                quantized_de = torch.cat((quantized_de, quantized_de[:, :, -8:]), dim=-1)

            quantized_de = model_vqvae.BN1d(quantized_de)

            # quantized_de size: (batch, 12, 5000)
            x_recon, low_feature, up_feature  = decoder_new(quantized_de.transpose(-1, -2))
            # low_feature: batch, 256, 313;  up_feature: batch, 5000, 64
    return low_feature

# def model_vqave_preprocess(data, model_vqvae):
#     with torch.no_grad():
#         z = model_vqvae._encoder(data) # z keys: 'last_hidden_state', 'loc', 'scale', 'patch_input'
#         # z.last_hidden_state: (batch, 12, patches, patch_length)
#         hidden_state = model_vqvae.BN(z.last_hidden_state)
#         quantization, indices, loss = model_vqvae._vq(hidden_state) # indices: (batch, 12, 624, 4)
#         # indices_all = indices.reshape(data.shape[0], 12, 624*4) / 1024
#         # feature_all = model_vqvae._vq.get_codes_from_indices(indices).permute(1,2,3,0,4) # feature_all: (batch, 12, 624, 4, 128)
#         # feature_all = feature_all.reshape(data.shape[0], 12, 624, 512)
#         # # 随机截取(data.shape[0], 12, 512, 512)的片段
#         # # start = np.random.randint(0, 112)
#         # return feature_all[:,:,:512,:]
#         quantization = (quantization.reshape(data.shape[0], 3, 4 , 624, 128)).permute(0, 1, 3, 4, 2)
#         quantization = quantization.reshape(data.shape[0], 3, 624, 512)
#     # # 删除不需要的变量同时释放空间
#     del z, hidden_state, indices, loss
#     torch.cuda.empty_cache()
#     return quantization[:,:,:512,:]
    
    
class ModelTrainer(BaseTrainer):

    def _train(self, iterator: Iterable) -> Tuple[float, float]:
        """
        Run train mode iteration.

        Args:
            iterator (Iterable):
        Returns:
            loss (float):
            score (float):
        """

        monitor = Monitor()
        self.model.train()

        for X, y in tqdm(iterator):
            # X: batch, 5000, 12, y: batch, 71
            X = X.float().to(self.device)
            y = y.float().to(self.device)
            
            # X = X.reshape(-1, 12, 2496)/1024
            # X = -torch.cos((X.reshape(-1, 12, 2496))/1023 * torch.pi)
            # X = X.permute(0, 2, 1)
            
            # X = model_vqave_preprocess(X, model_vqvae, decoder_new)
            # X = (X.reshape(X.size(0), 64, 4, 313)).reshape(X.size(0), 64, 4*313)
            X = X.permute(0, 2, 1)
            self.optimizer.zero_grad()
            y_pred = self.model(X) # X: batch, channel, feature
            minibatch_loss = self.loss_func(y_pred, y)
            minibatch_loss.backward()
            self.optimizer.step()

            monitor.store_loss(float(minibatch_loss), len(X))
            monitor.store_result(y, F.sigmoid(y_pred))
            # monitor.store_result(y, y_pred)

        loss = monitor.average_loss()
        score = monitor.macro_auc_roc()
        return loss, score

    def _evaluate(self, iterator: Iterable) -> Tuple[float, float]:
        """
        Run evaluation mode iteration.

        Args:
            iterator (Iterable):
        Returns:
            loss (float):
            score (float):
        """

        monitor = Monitor()
        self.model.eval()

        with torch.no_grad():
            for X, y in tqdm(iterator):
                X = X.float().to(self.device)
                y = y.float().to(self.device)
                # X = X.reshape(-1, 12, 2496)/1024
                # X = -torch.cos((X.reshape(-1, 12, 2496))/1023 * torch.pi)
                # X = X.permute(0, 2, 1)
                # y_pred = utils.aggregator(self.model, X)
                
                # X = model_vqave_preprocess(X, model_vqvae, decoder_new)
                # X = (X.reshape(X.size(0), 64, 4, 313)).reshape(X.size(0), 64, 4*313)
                X = X.permute(0, 2, 1)
                y_pred = self.model(X)

                monitor.store_loss(0, len(X))
                monitor.store_result(y, F.sigmoid(y_pred))
                # monitor.store_result(y, y_pred)

        loss = monitor.average_loss()
        score = monitor.macro_auc_roc()
        return loss, score

    def run(self, train_loader: Iterable, valid_loader: Iterable) -> None:
        """
        Args:
            train_loader (Iterable): Dataloader for training data.
            valid_loader (Iterable): Dataloader for validation data.
        Returns:
            None
        """

        # best_loss = np.inf # Sufficietly large
        # early_stopper = utils.EarlyStopper(mode="min", self.patience)

        best_score = -1 * np.inf # Sufficiently small
        early_stopper = utils.EarlyStopper(mode="max", patience=self.patience)
        writer = SummaryWriter(self.log_dir)

        for epoch in range(1, self.epochs+1):
            print("-"*80)
            print(f"Epoch {epoch}")
            train_loss, train_score = self._train(train_loader)
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_auc_roc", train_score, epoch)
            # print(f'-> Train loss: {train_loss:.4f}, score: {train_score:.4f}')
            print(f"Train loss: {train_loss:.4f}, macro_auc_roc: {train_score:.4f}")

            if epoch % self.eval_every == 0:
                eval_loss, eval_score = self._evaluate(valid_loader)
                # writer.add_scalar("eval_loss", eval_loss, epoch)
                writer.add_scalar("eval_auc_roc", eval_score, epoch)
                # print(f'-> Eval loss: {eval_loss:.4f}, score: {eval_score:.4f}')
                print(f"Eval macro_auc_roc: {eval_score:.4f}")

                # if eval_loss < best_loss:
                    # print(f"Validation loss improved {best_loss:.4f} -> {eval_loss:.4f}")
                    # best_loss = eval_loss
                    # self._save_model()
                if eval_score > best_score:
                    # print(f"Validation score improved {best_score:.4f} -> {eval_score:.4f}")
                    print(f"Current best test score: {eval_score:.4f}, before: {best_score:.4f}")
                    best_score = eval_score
                    self._save_model(best_score)

                if early_stopper.stop_training(eval_score):
                    print("Early stopping applied, stop training")
                    break
        print("-"*80)

class ModelTrainerMC(ModelTrainer):

    def set_lossfunc(self, weight:Optional[np.ndarray]=None) -> None:
        """
        Set loss function.
        Args:
            weight (Optional[np.ndarray]):
        Returns:
            None
        """
        if weight is not None:
            weight = torch.Tensor(weight).to(self.device)
        self.loss_func = nn.CrossEntropyLoss(weight=weight, reduction="sum")

    def _train(self, iterator: Iterable) -> Tuple[float, float]:
        """
        Run train mode iteration.

        Args:
            iterator (Iterable):
        Returns:
            loss (float):
            score (float):
        """

        monitor = Monitor()
        self.model.train()

        for X, y in tqdm(iterator):
            self.optimizer.zero_grad()
            X = X.float().to(self.device)
            y = y.long().to(self.device)
            y_pred = self.model(X)

            minibatch_loss = self.loss_func(y_pred, y)
            minibatch_loss.backward()
            self.optimizer.step()

            monitor.store_loss(float(minibatch_loss), len(X))
            monitor.store_result(y, F.softmax(y_pred))
            # monitor.store_result(y, y_pred)

        loss = monitor.average_loss()
        score = monitor.macro_f1()
        return loss, score

    def _evaluate(self, iterator: Iterable) -> Tuple[float, float]:
        """
        Run evaluation mode iteration.

        Args:
            iterator (Iterable):
        Returns:
            loss (float):
            score (float):
        """

        monitor = Monitor()
        self.model.eval()

        with torch.no_grad():
            for X, y in tqdm(iterator):
                X = X.float().to(self.device)
                y = y.long().to(self.device)

                y_pred = utils.aggregator(self.model, X)
                monitor.store_loss(0, len(X))
                monitor.store_result(y, F.softmax(y_pred))
                # monitor.store_result(y, y_pred)

        loss = monitor.average_loss()
        score = monitor.macro_f1()
        return loss, score
