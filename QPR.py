'''
Some modules refere to:
[1] https://github.com/yuqinie98/PatchTST
[2] https://github.com/lucidrains/vector-quantize-pytorch
[3] https://github.com/DeepPSP/torch_ecg
'''
import os
import pdb
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.utils import Config
# from positional_encodings.torch_encodings import PositionalEncoding1D, Summer # (batch, feature, channel)

from classification.supports.monitor import Monitor
from classification.classifier import model_clf
from encoder.mask_patch_self_attention import Musk_Patch_Attention, Musk_Patch_Attention_Config
from encoder.segmentor import find_wave_region
from encoder.component.mlp import MLP
from encoder.S4 import S4

from vector_quantize_pytorch import VectorQuantize, ResidualVQ, ResidualSimVQ, SimVQ
import utils.utils as utils


class qpr(nn.Module):
    '''
    Args:
    There are 2 kinds of params: about pacth encoder and about VQ
    patch:
    num_input_channels: input channel, consider 12
    context_length: input length, consider 5000 (or 1000)
    patch_length: patch length, consider 16
    patch_stride: patch stride, consider 8
    VQ:
    num_embeddings: number of codebook
    embedding_dim: dimension of codebook
    vq_form: different quantization method, 'normal', 'residual', 'simvq', 'residualsim'
    vq_heads: number of quantizers
    '''
    def __init__(self, 
                 num_input_channeles,
                 context_length,
                 patch_length,
                 patch_stride,
                 num_embeddings, 
                 embedding_dim, 
                 vq_form,
                 vq_heads,
                 multi_proj = False,
                 ):
        super(qpr, self).__init__()

        # patch encoder parameters
        self.num_input_channeles = num_input_channeles
        self.context_length = context_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        
        # vq parameters
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.vq_form = vq_form
        self.vq_heads = vq_heads
        self.config = Musk_Patch_Attention_Config(num_input_channels=self.num_input_channeles,
                                                  context_length=self.context_length,
                                                  patch_length=self.patch_length,
                                                  patch_stride=self.patch_stride,
                                                  channel_attention = True,
                                                  do_mask_input = True,
                                                  channel_consistent_masking = False, # mask all channels on the same position or not
                                                  mask_type = 'random',
                                                  random_mask_ratio = 0.4,
                                                  num_forecast_mask_patches = 124,
                                                  mask_value = 0, 
                                                  use_cls_token=False, # the 1st postion
                                                  )
        
        self._encoder = Musk_Patch_Attention(self.config)
  
        self._vq = self.get_vq(self.vq_form)

        self._decoder = S4(d_input = 12, d_model = 64)

        self.BN = nn.BatchNorm2d(12)    
        self.BN1d = nn.BatchNorm1d(12)
        
        self._map = nn.Sequential(
            MLP(input_dim = 128, output_dim = 64, activation='linear',n_activations=1),
            MLP(input_dim = 64, output_dim = 8, activation='linear',n_activations=1)
        )
        
        self._projection = MLP(input_dim = 64, output_dim = 12, activation='linear',n_activations=1)
        

    def get_vq(self, vq_form):
        # we offer different vq method, for ablation and comparison
        if vq_form == 'normal': 
            # need to reshape input size to (batch, 12*num_patches, 128)
            vq = VectorQuantize(dim=self.embedding_dim,
                                codebook_size = self.num_embeddings, 
                                decay = 0.8,
                                commitment_weight = 0.5,
                                kmeans_init = False,
                                kmeans_iters = 10,
                                rotation_trick = False
                                )
        elif vq_form == 'residual':
            vq = ResidualVQ(dim = self.embedding_dim,
                            num_quantizers = self.vq_heads, # residual quantization
                            codebook_size = self.num_embeddings,
                            stochastic_sample_codes = True,
                            sample_codebook_temp = 0.1, 
                            shared_codebook = True)
            
        elif vq_form == 'simvq':
            vq = SimVQ(dim = self.embedding_dim,
                       codebook_size = self.num_embeddings,
                       rotation_trick = True)

        elif vq_form == 'residualsim':
            vq = ResidualSimVQ(dim = self.embedding_dim,
                               num_quantizers = self.vq_heads,
                               codebook_size = self.num_embeddings,
                               rotation_trick = True,
                               commitment_weight = 0.5,
                               )
        else:
            raise ValueError('Load vq error, choose a suitable vq.')
        return vq
        

    def perplexity_cal(self, indices, num_embeddings, heads):
        perplexity_list = []
        # for multi-heads vq, we print the perplexity of each head
        if heads:
            for i in range(heads):
                ind = indices[:, :, i].reshape(-1)
                usage_count = torch.bincount(ind, minlength = num_embeddings)
                total_count = ind.numel()
                probs = usage_count.float() / total_count
                nonzero_probs = probs[probs > 0]
                perplexity = torch.exp(-torch.sum(nonzero_probs * torch.log(nonzero_probs)))
                perplexity_list.append(perplexity)
            return perplexity_list
        else:
            ind = indices.reshape(-1)
            usage_count = torch.bincount(ind, minlength = num_embeddings)
            total_count = ind.numel()
            probs = usage_count.float() / total_count
            nonzero_probs = probs[probs > 0]
            perplexity = torch.exp(-torch.sum(nonzero_probs * torch.log(nonzero_probs)))
            return perplexity


    def forward(self, x):
        # input x: (batch, 5000, 12)
        z = self._encoder(x) # z keys: 'last_hidden_state', 'loc', 'scale', 'patch_input'
        
        # z.last_hidden_state: (batch, 12, patches, patch_length)
        hidden_state = self.BN(z.last_hidden_state)

        quantization, indices, loss = self._vq(hidden_state) # (batch, 12, 625/500, 128)
        
        quantized_de = self._map(quantization).reshape(x.shape[0], 12, -1) # batch, 12, 5000
        
        if quantized_de.shape[-1] < 5000:
            quantized_de = torch.cat((quantized_de, quantized_de[:, :, -8:]), dim=-1)
        
        quantized_de = self.BN1d(quantized_de)
        
        # quantized_de size: (batch, 12, 5000)
        x_recon = self._decoder(quantized_de.transpose(-1, -2))
        # x_recon size: (batch, 5000, 64)

        x_recon = self._projection(x_recon)

        x_recon = x_recon * z.scale + z.loc # destandardization

        perplexity = self.perplexity_cal(indices, self.num_embeddings, self.vq_heads)
        loss = loss.sum() / self.vq_heads

        return loss, x_recon, quantization, indices, perplexity
    
    
def get_optimizer(optimizer_name, model, learning_rate):
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
            amsgrad=False
        )
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer name: {optimizer_name}")
    return optimizer


# class _decoder_clf(nn.Module):
#     def __init__(self):
#         super(_decoder_clf, self).__init__()
        
#         self._in_s4 = S4(d_input = 12, d_model = 64)
        
#         self._lower_conv = nn.Sequential(
#             nn.BatchNorm1d(64),
#             nn.Conv1d(in_channels = 64, out_channels = 96, kernel_size = 5, stride = 2, padding = 1),
#             nn.Conv1d(in_channels = 96, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
#             nn.Conv1d(in_channels = 128, out_channels = 192, kernel_size = 3, stride = 2, padding = 1),
#             nn.Conv1d(in_channels = 192, out_channels = 256, kernel_size = 3, stride = 2, padding = 1)
#             )
        
#         self._upper_conv  = nn.Sequential(
#             nn.ConvTranspose1d(in_channels = 256, out_channels = 192, kernel_size = 3, stride = 2, padding = 1),
#             nn.ConvTranspose1d(in_channels = 192, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
#             nn.ConvTranspose1d(in_channels = 128, out_channels = 96, kernel_size = 3, stride = 2),
#             nn.ConvTranspose1d(in_channels = 96, out_channels = 64, kernel_size = 3, stride = 2, output_padding=1),
#             )
        
#         self._out_s4 = S4(d_input = 64, d_model = 12)
        
#         self._proj_clf = MLP(input_dim = 12, output_dim = 12, activation='linear',n_activations=1)

#     def forward(self, x):
#         assert x.size(-1) == 12
#         x = self._in_s4(x)
        
#         x = x.transpose(-1, -2)
#         low_feature = self._lower_conv(x) # batch, 256, 313
#         up_feature = self._upper_conv(low_feature)
#         up_feature = up_feature.transpose(-1, -2) # batch, 5000, 64
  
#         output = self._out_s4(up_feature)     
#         output = self._proj_clf(output)
#         return output, low_feature, up_feature


# In pre-training, we use 3 chest leads to generate the entire 6 leads, with linear rotation method. 
# (use or not donot affect the result)
class LeadTransformer(nn.Module):
    # init 3 persudo leads, then use the 3 leads to generate 6 chest leads
    def __init__(self, signal_length=5000):
        super(LeadTransformer, self).__init__()
        
        # self.num_leads = num_leads
        self.signal_length = signal_length
        
        self.angles = nn.ParameterList([nn.Parameter(torch.randn(self.signal_length)) for _ in range(6)])
        self.linear_weights = nn.ParameterList([nn.Parameter(torch.randn(self.signal_length)) for _ in range(6)]) 
        self.linear_biases = nn.ParameterList([nn.Parameter(torch.randn(self.signal_length)) for _ in range(6)])

        self.map = MLP(input_dim = 6, output_dim = 3, activation='linear', n_activations=1)

    def forward(self, given_lead):
        """
        given_lead: persudo leads, size: (batch, 5000, 6), we init from chest leads(MLP process)
        """
        given_lead = self.map(given_lead)
        transformed_leads = []
        for i in range(6):
            lead_index = i // 2
            transformed_lead = self.rotation_proj(given_lead[:, :, lead_index], self.angles[i], self.linear_weights[i], self.linear_biases[i])
            transformed_leads.append(transformed_lead)
        
        transformed_leads = torch.cat(transformed_leads, dim=2)

        return transformed_leads

    def rotation_proj(self, base_lead, angles, linear_weights, linear_biases):
        batch_size = base_lead.size(0)

        # rotation matrix
        cos_angles = torch.cos(angles).view(1, 1, -1)  # cos
        sin_angles = torch.sin(angles).view(1, 1, -1)  # sin
        
        # angle transformation for each lead
        rotated_leads = cos_angles * base_lead.unsqueeze(1) + \
                        sin_angles * torch.roll(base_lead.unsqueeze(1), shifts=1, dims=2)

        # linear transformation for each lead
        transformed_leads = linear_weights.view(1, 1, -1) * rotated_leads + \
                            linear_biases.view(1, 1, -1)
                            
        transformed_leads = transformed_leads.view(batch_size, self.signal_length, -1)
        return transformed_leads



# train qpr on mimic
def train_vq(model_vqvae, 
             lead_transformer, 
             train_dataloader, 
             lr, 
             epoch, 
             device, 
             aug_gaussian = False, 
             Einthoven_loss=False, 
             log_path=None, 
             save_model_dir = None,
             checkpoint_path=None):
    
    train_res_recon_error = []
    train_vq_loss = []
    train_res_perplexity = {}
    train_einthoven_error = []
    train_transform_loss = []
    
    vqvae_optimizer = get_optimizer('adamw', model_vqvae, lr)
    lead_transformer_optimizer = get_optimizer('adamw', lead_transformer, lr)
    
    model_vqvae.train()
    lead_transformer.train()
    
    # Tensorboard writer
    writer = SummaryWriter(log_path)
    
    # amp initialize
    # scaler = amp.GradScaler()
    
    # init number of epoch and batch that processed
    start_epoch = 0
    start_batch = 0

    # load checkpoint with model states and epoch-batch, while using 'continue' to skip nan loss
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model_vqvae.load_state_dict(checkpoint['model_vqvae_state_dict'])
        lead_transformer.load_state_dict(checkpoint['lead_transformer_state_dict'])
        vqvae_optimizer.load_state_dict(checkpoint['vqvae_optimizer_state_dict'])
        lead_transformer_optimizer.load_state_dict(checkpoint['lead_transformer_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch} batch {start_batch}")

    for step in tqdm(range(start_epoch, epoch), desc="Pre-Training Progress"):

        train_dataloader_iter = iter(train_dataloader)
        cal_num = 0

        # ignore the batches trained before
        for _ in range(start_batch):
            next(train_dataloader_iter)
            cal_num += 1

        for data_batch in train_dataloader_iter:
            cal_num += 1

            data = data_batch.float()
            data = data.to(device)
            
            if aug_gaussian:
                data = utils.add_noise(data, noise_level=0.01)

            vqvae_optimizer.zero_grad()
            lead_transformer_optimizer.zero_grad()
            
            # with amp.autocast():
            
            vq_loss, x_recon, quantization, indices, perplexities = model_vqvae(data)
            
            # chest-leads transformation loss
            transformed_leads = lead_transformer(x_recon[:, :, 6:])
            transform_loss  = F.mse_loss(transformed_leads, data[:, :, 6:])
            
            # reconstruct loss
            recon_error = F.mse_loss(x_recon, data)
            
            if Einthoven_loss:
                T_einthoven_list = [
                    utils.T_einthoven_12all.to(device),
                    utils.T_einthoven_13all.to(device),
                    utils.T_einthoven_23all.to(device)
                ]
                einthoven_errors = [
                    F.mse_loss(x_recon @ T_einthoven, data @ T_einthoven)
                    for T_einthoven in T_einthoven_list
                ]
                einthoven_error = sum(einthoven_errors) / 3
                loss = recon_error + vq_loss + einthoven_error + 0.25*transform_loss
            else:
                loss = recon_error + vq_loss + 0.25*transform_loss
            
            # check NaN
            if torch.isnan(loss):
                print(f"Iteration {step+1}: Loss is NaN")
                print(f"Dataloader number: {cal_num}")
                print()
                del data_batch
                del data
                del x_recon
                del quantization
                del indices
                del perplexities
                torch.cuda.empty_cache()
                print('clear over!')
                continue

            loss.backward()
            vqvae_optimizer.step()
            lead_transformer_optimizer.step()
            
            # scaler.scale(loss).backward()
            # scaler.step(vqvae_optimizer)
            # scaler.step(lead_transformer_optimizer)
            # scaler.update()

            train_res_recon_error.append(recon_error.item())
            train_vq_loss.append(vq_loss.item())
            train_transform_loss.append(transform_loss.item())
            
            for i, perplexity in enumerate(perplexities):
                if i not in train_res_perplexity:
                    train_res_perplexity[i] = []
                train_res_perplexity[i].append(perplexity.item())
            
            # train_res_perplexity.append(perplexity.item())
            if Einthoven_loss:
                train_einthoven_error.append(einthoven_error.item())

            if (step+1) % 1 == 0 and cal_num % 600 == 0:
                writer.add_scalar('recon_error', np.mean(train_res_recon_error[-100:]), (step+1))
                writer.add_scalar('vq_loss', np.mean(train_vq_loss[-100:]), (step+1))
                # writer.add_scalar('learning_rate', vqvae_optimizer.param_groups[0]['lr'], (step+1))
                writer.add_scalar('transform_loss', np.mean(train_transform_loss[-100:]), (step+1))
                
                # writer.add_scalar('perplexity', np.mean(train_res_perplexity[-100:]), (step+1))
                
                print('%d iterations' % (step+1))
                print('%d batches in train_dataloader' % cal_num)
                print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
                print('vq_loss: %.3f' % np.mean(train_vq_loss[-100:]))
                # print(f'prediction loss print: {F.mse_loss(x_recon[:, -1000:,:], data[:, -1000:,:])}')
                if Einthoven_loss:
                    writer.add_scalar('einthoven_loss', np.mean(train_einthoven_error[-100:]), (step+1))
                    print('einthoven_loss: %.3f' % np.mean(train_einthoven_error[-100:]))
                print('transform_loss: %.3f' % np.mean(train_transform_loss[-100:]))
                    
                # print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:])) 
                for i, perplexity_list in train_res_perplexity.items():
                    avg_perplexity = np.mean(perplexity_list[-100:])
                    writer.add_scalar(f'perplexity_head_{i}', avg_perplexity, (step + 1))
                    print(f'perplexity (head {i}): {avg_perplexity:.3f}')
                    
                print()
            
            # save checkpoint
            if cal_num % 13500 == 0 and checkpoint_path is not None:
                torch.save({
                    'epoch': step,
                    'batch': cal_num,
                    'model_vqvae_state_dict': model_vqvae.state_dict(),
                    'lead_transformer_state_dict': lead_transformer.state_dict(),
                    'vqvae_optimizer_state_dict': vqvae_optimizer.state_dict(),
                    'lead_transformer_optimizer_state_dict': lead_transformer_optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {step+1} batch {cal_num}")

        # after process one epoch, set start_batch to 0
        start_batch = 0
                        
        if (step + 1) % 2 == 0 and save_model_dir is not None:
            torch.save(model_vqvae, f"{save_model_dir}/vqvae_iter_{step+1}.pth")
            torch.save(lead_transformer, f"{save_model_dir}/lead_transformer_iter_{step+1}.pth")

    
    writer.close()
    if save_model_dir is not None:
        torch.save(model_vqvae, f'{save_model_dir}/vqvae_final.pth')
        torch.save(lead_transformer, f'{save_model_dir}/lead_transformer_final.pth')



def finetune_vq(signal_length,
                model_vqvae, 
                lead_transformer, 
                predictionHead,
                train_dataloader, 
                lr, 
                epoch, 
                device, 
                Einthoven_loss=False, 
                log_path=None, 
                save_model_dir = None,
                checkpoint_path=None):
    
    train_res_recon_error = []
    train_mask_mae = []
    train_mask_rmse = []
    train_vq_loss = []
    train_res_perplexity = {}
    train_einthoven_error = []
    train_transform_loss = []
    
    vqvae_optimizer = get_optimizer('adamw', model_vqvae, lr)
    lead_transformer_optimizer = get_optimizer('adamw', lead_transformer, lr)
    
    model_vqvae.train()
    lead_transformer.train()
    
    # Tensorboard writer
    writer = SummaryWriter(log_path)
    
    # amp initialize
    # scaler = amp.GradScaler()
    
    # 初始化已处理的epoch和batch数
    start_epoch = 0
    start_batch = 0

    # load checkpoint with model states and epoch-batch, while using 'continue' to skip nan loss
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model_vqvae.load_state_dict(checkpoint['model_vqvae_state_dict'])
        lead_transformer.load_state_dict(checkpoint['lead_transformer_state_dict'])
        # predictionHead.load_state_dict(checkpoint['predictionHead_state_dict'])
        vqvae_optimizer.load_state_dict(checkpoint['vqvae_optimizer_state_dict'])
        lead_transformer_optimizer.load_state_dict(checkpoint['lead_transformer_optimizer_state_dict'])
        # predictionHead_optimizer.load_state_dict(checkpoint['predictionHead_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch} batch {start_batch}")

    for step in tqdm(range(start_epoch, epoch), desc="Fine-tuning Progress"):

        train_dataloader_iter = iter(train_dataloader)
        cal_num = 0

        # ignore the batches trained before
        for _ in range(start_batch):
            next(train_dataloader_iter)
            cal_num += 1

        for data_batch in train_dataloader_iter:
            cal_num += 1

            (data, _) = data_batch
            data = data.float()
            if data.shape[-1] == signal_length:
                data = data.permute(0, 2, 1)    
            data = data.to(device)
                               

            vqvae_optimizer.zero_grad()
            lead_transformer_optimizer.zero_grad()
            # predictionHead_optimizer.zero_grad()
            
            # with amp.autocast():
            vq_loss, x_recon, quantization, indices, perplexities = model_vqvae(data)
            transformed_leads = lead_transformer(x_recon[:, :, 6:])
            
            # chest-leads transformation loss and reconstruct loss and einthoven loss
            transform_loss  = F.mse_loss(transformed_leads, data[:, :, 6:])
            recon_error = F.mse_loss(x_recon, data)
            
            if Einthoven_loss:
                T_einthoven_list = [
                    utils.T_einthoven_12all.to(device),
                    utils.T_einthoven_13all.to(device),
                    utils.T_einthoven_23all.to(device)
                ]
                einthoven_errors = [
                    F.mse_loss(x_recon @ T_einthoven, data @ T_einthoven)
                    for T_einthoven in T_einthoven_list
                ]
                einthoven_error = sum(einthoven_errors) / 3
                loss = recon_error + vq_loss + einthoven_error + 0.25*transform_loss
            else:
                loss = recon_error + vq_loss + 0.25*transform_loss
            
            mask_ratio = model_vqvae._encoder.masking.random_mask_ratio
            if mask_ratio != 0:
                # about mask
                z = model_vqvae._encoder(data)
                mask = z.mask.unsqueeze(-1).repeat(1, 1, 1, 16) # (batch, 12, 625, 16) 
                patchifier = model_vqvae._encoder.patchifier
                
                original_patches = patchifier(data).masked_fill(~mask.bool(), 0)
                recon_patches = patchifier(x_recon).masked_fill(~mask.bool(), 0)
                
                prediction_mae = F.l1_loss(recon_patches, original_patches)
                prediction_mse = torch.sqrt(F.mse_loss(recon_patches, original_patches))

            # 检查损失是否为 NaN
            if torch.isnan(loss):
                print(f"Iteration {step+1}: Loss is NaN")
                print(f"Dataloader number: {cal_num}")
                print()
                del data_batch
                del data
                del x_recon
                del quantization
                del indices
                del perplexities
                torch.cuda.empty_cache()
                print('clear over!')
                continue

            loss.backward()
            vqvae_optimizer.step()
            lead_transformer_optimizer.step()
            
            # scaler.scale(loss).backward()
            # scaler.step(vqvae_optimizer)
            # scaler.step(lead_transformer_optimizer)
            # scaler.update()

            train_res_recon_error.append(recon_error.item())
            if mask_ratio != 0:
                train_mask_mae.append(prediction_mae.item())
                train_mask_rmse.append(prediction_mse.item())
            train_vq_loss.append(vq_loss.item())
            train_transform_loss.append(transform_loss.item())
            
            for i, perplexity in enumerate(perplexities):
                if i not in train_res_perplexity:
                    train_res_perplexity[i] = []
                train_res_perplexity[i].append(perplexity.item())

            if Einthoven_loss:
                train_einthoven_error.append(einthoven_error.item())

            if (step+1) % 1 == 0 and cal_num % 5 == 0:
                writer.add_scalar('recon_error', np.mean(train_res_recon_error[-100:]), (step+1))
                if mask_ratio != 0:
                    writer.add_scalar('mask_mae', np.mean(train_mask_mae[-100:]), (step+1))
                    writer.add_scalar('mask_rmse', np.mean(train_mask_rmse[-100:]), (step+1))
                writer.add_scalar('vq_loss', np.mean(train_vq_loss[-100:]), (step+1))
                writer.add_scalar('transform_loss', np.mean(train_transform_loss[-100:]), (step+1))
                
                print('%d iterations' % (step+1))
                print('%d batches in train_dataloader' % cal_num)
                print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
                if mask_ratio != 0:
                    print('mask_mae: %.6f' % np.mean(train_mask_mae[-100:]))
                    print('mask_rmse: %.6f' % np.mean(train_mask_rmse[-100:]))
                print('vq_loss: %.3f' % np.mean(train_vq_loss[-100:]))
                if Einthoven_loss:
                    writer.add_scalar('einthoven_loss', np.mean(train_einthoven_error[-100:]), (step+1))
                    print('einthoven_loss: %.3f' % np.mean(train_einthoven_error[-100:]))
                print('transform_loss: %.3f' % np.mean(train_transform_loss[-100:]))

                for i, perplexity_list in train_res_perplexity.items():
                    avg_perplexity = np.mean(perplexity_list[-100:])
                    writer.add_scalar(f'perplexity_head_{i}', avg_perplexity, (step + 1))
                    print(f'perplexity (head {i}): {avg_perplexity:.3f}')
                    
                print()
            
            # save checkpoint
            if cal_num % 2000 == 0 and checkpoint_path is not None:
                torch.save({
                    'epoch': step,
                    'batch': cal_num,
                    'model_vqvae_state_dict': model_vqvae.state_dict(),
                    'lead_transformer_state_dict': lead_transformer.state_dict(),
                    'vqvae_optimizer_state_dict': vqvae_optimizer.state_dict(),
                    'lead_transformer_optimizer_state_dict': lead_transformer_optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {step+1} batch {cal_num}")

        start_batch = 0
                        
        if (step + 1) % 15 == 0:
            if not os.path.exists(save_model_dir):
                os.makedirs(save_model_dir)
            torch.save(model_vqvae, f"{save_model_dir}/vqvae_iter_{step+1}.pth")
            torch.save(lead_transformer, f"{save_model_dir}/lead_transformer_iter_{step+1}.pth")
    
    writer.close()
    torch.save(model_vqvae, f'{save_model_dir}/vqvae_final.pth')
    torch.save(lead_transformer, f'{save_model_dir}/lead_transformer_final.pth')


# def prediction_vq(model_vqvae, 
#                   model_pred,
#                   train_dataloader, 
#                   test_dataloader,
#                   lr, 
#                   epochs, 
#                   device, 
#                   log_path=None, 
#                   save_model_dir = None): # modified!!

#     train_indices_loss = []
#     train_recon_loss = []
#     train_vq_loss = []
    
#     model_vqvae_optimizer = get_optimizer('adamw', model_vqvae, lr)
#     model_pred_optimizer = get_optimizer('adam', model_pred, lr)
    
#     model_vqvae.train()
#     model_pred.train()

#     writer = SummaryWriter(log_path)
    
#     # amp initialize
#     # scaler = amp.GradScaler()

#     for epoch in range(epochs):

#         for batch_idx, (data, _) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            
#             # data = data.float()
#             data = data.to(device) # b, 1000/5000, 12

#             model_pred_optimizer.zero_grad()
#             model_vqvae_optimizer.zero_grad()
            
#             # with torch.no_grad():
#             # vq_loss, x_recon, quantization, indices, perplexities = model_vqvae(data)
#             z = model_vqvae._encoder(data) # z keys: 'last_hidden_state', 'loc', 'scale', 'patch_input'
            
#             # z.last_hidden_state: (batch, 12, patches, patch_length)
#             hidden_state = model_vqvae.BN(z.last_hidden_state)

#             quantization, indices, vq_loss = model_vqvae._vq(hidden_state) # (batch, 12, 625/500, 128)
            
#             # with amp.autocast():
#             # prediction, vq_loss, perplexities = model_vqvae(data)
            
#             inputs = indices[:, :, :561]
#             targets = indices[:, :, 561:]
            
#             # 前向传播
#             logits = model_pred(inputs)  # (batch, 12, 99, 4, codebook_size)
            
#             # 计算损失（只计算预测部分的loss）
#             pred_loss = 0
#             for t in range(63):  # 预测63个时间步
#                 # 每个时间步对应不同码本的预测
#                 for codebook in range(4):
#                     pred = logits[:, :, -63+t, codebook]  # 逐步展开预测
#                     target = targets[:, :, t, codebook]
#                     pred_loss += nn.CrossEntropyLoss()(pred.view(-1, 1024), target.view(-1))
            
#             pred_loss /= (63 * 4)  # 平均损失
            
#             pred_indices = model_pred.predict(inputs, predict_steps=63)
#             indices_all = torch.cat((inputs, pred_indices), dim=2)
#             quantization = model_vqvae._vq.get_output_from_indices(indices_all)
#             # quantization = quantization.detach().requires_grad_(True)
            
#             quantized_de = model_vqvae._map(quantization).reshape(data.shape[0], 12, -1) # batch, 12, 5000
            
#             if quantized_de.shape[-1] < 5000:
#                 quantized_de = torch.cat((quantized_de, quantized_de[:, :, -8:]), dim=-1)
            
#             quantized_de = model_vqvae.BN1d(quantized_de)
            
#             # quantized_de size: (batch, 12, 5000)
#             x_recon = model_vqvae._decoder(quantized_de.transpose(-1, -2))
#             # x_recon size: (batch, 5000, 64)

#             x_recon = model_vqvae._projection(x_recon)

#             x_recon = x_recon * z.scale + z.loc # destandardization

#             # perplexity = model_vqvae.perplexity_cal(indices, model_vqvae.num_embeddings, model_vqvae.vq_heads)
#             vq_loss = vq_loss.sum() / model_vqvae.vq_heads
            
#             recon_loss = F.mse_loss(x_recon[:, -500:, :], data[:, -500:, :])
            
#             loss = pred_loss + vq_loss + recon_loss

#             loss.backward()
#             model_vqvae_optimizer.step()
#             model_pred_optimizer.step()            
#             # scaler.scale(loss).backward()
#             # scaler.step(model_vqvae_optimizer)
#             # scaler.step(model_pred_optimizer)
#             # scaler.update()

#             train_indices_loss.append(loss.item())
#             train_recon_loss.append(recon_loss.item())
#             train_vq_loss.append(vq_loss.item())

#             if (batch_idx+1) % 20 == 0:
#                 writer.add_scalar('indices_loss', np.mean(train_indices_loss[-100:]), (epoch+1))
#                 writer.add_scalar('recon_loss', np.mean(train_recon_loss[-100:]), (epoch+1))
#                 writer.add_scalar('vq_loss', np.mean(train_vq_loss[-100:]), (epoch+1))
#                 print('%d iterations' % (epoch+1))
#                 print('indices_loss: %.5f' % np.mean(train_indices_loss[-100:]))
#                 print('recon_loss: %.5f' % np.mean(train_recon_loss[-100:]))
#                 print('vq_loss: %.5f' % np.mean(train_vq_loss[-100:]))
#                 print()
        
#         if (epoch + 1) % 1 == 0:
#             test_indics_loss = []
#             test_recon_loss = []
#             test_vq_loss = []

#             for (data, _) in test_dataloader:
#                 model_vqvae.eval()
#                 model_pred.eval()
#                 with torch.no_grad():
#                     data = data.to(device)
#                     z = model_vqvae._encoder(data) # z keys: 'last_hidden_state', 'loc', 'scale', 'patch_input'
                    
#                     # z.last_hidden_state: (batch, 12, patches, patch_length)
#                     hidden_state = model_vqvae.BN(z.last_hidden_state)

#                     _, indices, vq_loss = model_vqvae._vq(hidden_state) # (batch, 12, 625/500, 128)
                    
#                     # with amp.autocast():
#                     # prediction, vq_loss, perplexities = model_vqvae(data)
                    
#                     # 输入数据格式: (batch, 12, 124, 4) 前99为输入，后25为target
#                     inputs = indices[:, :, :561]   # (batch, 12, 99, 4)
#                     targets = indices[:, :, 561:]  # (batch, 12, 25, 4)
                    
#                     # 前向传播
#                     logits = model_pred(inputs)  # (batch, 12, 99, 4, codebook_size)
                    
#                     # 计算损失（只计算预测部分的loss）
#                     pred_loss = 0
#                     for t in range(63):  # 预测63个时间步
#                         # 每个时间步对应不同码本的预测
#                         for codebook in range(4):
#                             pred = logits[:, :, -63+t, codebook]  # 逐步展开预测
#                             target = targets[:, :, t, codebook]
#                             pred_loss += nn.CrossEntropyLoss()(pred.view(-1, 1024), target.view(-1))
                    
#                     pred_loss /= (63 * 4)  # 平均损失
                    
#                     pred_indices = model_pred.predict(inputs, predict_steps=63)
                    
#                     indices_all = torch.cat((inputs, pred_indices), dim=2)
                    
#                     quantization = model_vqvae._vq.get_output_from_indices(indices_all)
                    
#                     quantized_de = model_vqvae._map(quantization).reshape(data.shape[0], 12, -1) # batch, 12, 5000
                    
#                     if quantized_de.shape[-1] < 5000:
#                         quantized_de = torch.cat((quantized_de, quantized_de[:, :, -8:]), dim=-1)
                    
#                     quantized_de = model_vqvae.BN1d(quantized_de)
                    
#                     # quantized_de size: (batch, 12, 5000)
#                     x_recon = model_vqvae._decoder(quantized_de.transpose(-1, -2))
#                     # x_recon size: (batch, 5000, 64)

#                     x_recon = model_vqvae._projection(x_recon)

#                     x_recon = x_recon * z.scale + z.loc # destandardization

#                     perplexity = model_vqvae.perplexity_cal(indices, model_vqvae.num_embeddings, model_vqvae.vq_heads)
#                     vq_loss = vq_loss.sum() / model_vqvae.vq_heads
                    
#                     recon_loss = F.mse_loss(x_recon[:, -500:, :], data[:, -500:, :])
                    
#                     # loss = pred_loss + vq_loss + recon_loss

#                     test_indics_loss.append(pred_loss.item())
#                     test_recon_loss.append(recon_loss.item())
#                     test_vq_loss.append(vq_loss.item())
#             writer.add_scalar('Test indices_loss', np.mean(test_indics_loss), (epoch+1))
#             writer.add_scalar('Test recon_loss', np.mean(test_recon_loss), (epoch+1))
#             writer.add_scalar('Test vq_loss', np.mean(test_vq_loss), (epoch+1))
#             print(f'Test indices_loss: {np.mean(test_indics_loss)}')
#             print(f'Test recon_loss: {np.mean(test_recon_loss)}')
#             print(f'Test vq_loss: {np.mean(test_vq_loss)}')
#             print()
            
                        
#         if (epoch + 1) % 1 == 0 and save_model_dir is not None:
#             if not os.path.exists(save_model_dir):
#                 os.makedirs(save_model_dir)
#             torch.save(model_vqvae, f"{save_model_dir}/vqvae_iter_{epoch+1}.pth")
#             torch.save(model_pred, f"{save_model_dir}/pred_iter_{epoch+1}.pth")
    
#     writer.close()
#     if save_model_dir is not None:
#         torch.save(model_vqvae, f'{save_model_dir}/vqvae_final.pth')
#         torch.save(model_pred, f'{save_model_dir}/pred_final.pth')


# def classification_MC(model_clf,
#                       train_dataloader, 
#                       test_dataloader,
#                       weight,
#                       lr, 
#                       epoch, 
#                       device, 
#                       log_path=None, 
#                       save_model_dir = None,
#                       checkpoint_path=None):
    
#     train_res_recon_error = []
#     train_vq_loss = []
#     train_res_perplexity = {}
    
#     model_clf_optimizer = get_optimizer('adamw', model_clf, lr)

#     writer = SummaryWriter(log_path)
    
#     # amp initialize
#     # scaler = amp.GradScaler()
    
#     # 初始化已处理的epoch和batch数
#     start_epoch = 0
#     start_batch = 0

#     # load checkpoint with model states and epoch-batch, while using 'continue' to skip nan loss
#     if checkpoint_path is not None and os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path)
#         model_clf.load_state_dict(checkpoint['model_vqvae_state_dict'])
#         model_clf_optimizer.load_state_dict(checkpoint['vqvae_optimizer_state_dict'])
#         start_epoch = checkpoint['epoch']
#         start_batch = checkpoint['batch']
#         print(f"Checkpoint loaded. Resuming from epoch {start_epoch} batch {start_batch}")


#     for step in tqdm(range(start_epoch, epoch), desc="Fine-tuning Progress"):
#         # full size dataset for pre-clf
#         train_dataloader_iter = iter(train_dataloader)
#         cal_num = 0
#         # ignore the batches trained before
#         for _ in range(start_batch):
#             next(train_dataloader_iter)
#             cal_num += 1

#         monitor = Monitor()
#         for data_batch in train_dataloader_iter:
#             cal_num += 1

#             model_clf.train()
#             (data, label) = data_batch
#             if data.shape[0] == 1:
#                 break
#             data = data.float()
#             data = data.to(device)
#             label = label.to(device)

#             model_clf_optimizer.zero_grad()
            
#             # with amp.autocast():
#             # input data: (batch, 5000, 12)
#             y_pred, vq_loss, perplexities = model_clf(data.permute(0, 2, 1))

#             weight = torch.Tensor(weight).to(device)
#             recon_error = nn.CrossEntropyLoss(weight=weight, reduction="sum")(y_pred, label)
            
#             loss = recon_error + vq_loss
            
#             # 检查损失是否为 NaN
#             if torch.isnan(loss):
#                 print(f"Iteration {step+1}: Loss is NaN")
#                 print(f"Dataloader number: {cal_num}")
#                 print()
#                 del data_batch
#                 del data
#                 del perplexities
#                 torch.cuda.empty_cache()
#                 print('clear over!')
#                 continue

#             loss.backward()
#             model_clf_optimizer.step()
#             monitor.store_result(label, F.softmax(y_pred, dim=-1))
            
#             # scaler.scale(loss).backward()
#             # scaler.step(vqvae_optimizer)
#             # scaler.step(lead_transformer_optimizer)
#             # scaler.update()

#             train_res_recon_error.append(recon_error.item())
#             train_vq_loss.append(vq_loss.item())
            
#             for i, perplexity in enumerate(perplexities):
#                 if i not in train_res_perplexity:
#                     train_res_perplexity[i] = []
#                 train_res_perplexity[i].append(perplexity.item())

#             if (step+1) % 1 == 0 and cal_num % 600 == 0:
#                 score = monitor.macro_f1()
#                 writer.add_scalar('recon_error', np.mean(train_res_recon_error), (step+1))
#                 writer.add_scalar('vq_loss', np.mean(train_vq_loss), (step+1))
#                 writer.add_scalar('macro_f1', score, (step+1))
                
#                 print('%d iterations' % (step+1))
#                 print('%d batches in train_dataloader' % cal_num)
#                 print('recon_error/bce loss: %.3f' % np.mean(train_res_recon_error))
#                 print('macro_f1auc_roc: %.5f' % score)
#                 print('vq_loss: %.3f' % np.mean(train_vq_loss))

#                 for i, perplexity_list in train_res_perplexity.items():
#                     avg_perplexity = np.mean(perplexity_list)
#                     writer.add_scalar(f'perplexity_head_{i}', avg_perplexity, (step + 1))
#                     print(f'perplexity (head {i}): {avg_perplexity:.3f}')
#                 print()
            
#             # save checkpoint
#             if (step+1) % 5 == 0 and cal_num % 2800 == 0 and checkpoint_path is not None:
#                 torch.save({
#                     'epoch': step,
#                     'batch': cal_num,
#                     'model_clf_state_dict': model_clf.state_dict(),
#                     'model_clf_optimizer_state_dict': model_clf_optimizer.state_dict(),
#                 }, checkpoint_path)
#                 print(f"Checkpoint saved at epoch {step+1} batch {cal_num}")

#         # 处理完一个epoch后，将start_batch置0
#         start_batch = 0
                        
#         if (step + 1) % 1 == 0:
#             test_label_error = []
#             monitor = Monitor()
#             for (data, label) in test_dataloader:
#                 model_clf.eval()
#                 with torch.no_grad():
#                     data = data.to(device)
#                     label = label.to(device)
#                     y_pred, _, _ = model_clf(data.permute(0, 2, 1))
#                     recon_error = nn.CrossEntropyLoss(weight=weight, reduction="sum")(y_pred, label)
#                     test_label_error.append(recon_error.item())
#                     monitor.store_result(label, F.softmax(y_pred, dim=-1))
#             score = monitor.macro_f1()
#             writer.add_scalar('Test BCE loss', np.mean(test_label_error), (step+1))
#             writer.add_scalar('Test macro_f1', score, (step+1))
#             print(f'Test BCE loss: {np.mean(test_label_error)}')
#             print(f'Test macro_f1: {score}')
        
#         if (step + 1) % 5 == 0 and save_model_dir is not None:
#             if not os.path.exists(save_model_dir):
#                 os.makedirs(save_model_dir)
#             torch.save(model_clf, f"{save_model_dir}/clf_mc_iter_{step+1}.pth")
            
#     writer.close()
#     if save_model_dir is not None:
#         torch.save(model_clf, f'{save_model_dir}/clf_mc_final.pth')


# multi-label classification
def classification_ML(model_clf,
                      train_dataloader, 
                      test_dataloader,
                      lr, 
                      epoch, 
                      device, 
                      Einthoven_loss=False, 
                      log_path=None, 
                      save_model_dir = None,
                      checkpoint_path=None):
    
    train_res_recon_error = []
    train_vq_loss = []
    train_res_perplexity = {}
    train_einthoven_error = []
    
    model_clf_optimizer = get_optimizer('adamw', model_clf, lr)
    

    writer = SummaryWriter(log_path)
    
    # amp initialize
    # scaler = amp.GradScaler()
    
    # 初始化已处理的epoch和batch数
    start_epoch = 0
    start_batch = 0

    # load checkpoint with model states and epoch-batch, while using 'continue' to skip nan loss
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model_clf.load_state_dict(checkpoint['model_vqvae_state_dict'])
        model_clf_optimizer.load_state_dict(checkpoint['vqvae_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch} batch {start_batch}")


    for step in tqdm(range(start_epoch, epoch), desc="Fine-tuning Progress"):
        # full size dataset for pre-clf
        train_dataloader_iter = iter(train_dataloader)
        cal_num = 0
        # ignore the batches trained before
        for _ in range(start_batch):
            next(train_dataloader_iter)
            cal_num += 1

        monitor = Monitor()
        for data_batch in train_dataloader_iter:
            cal_num += 1

            model_clf.train()
            (data, label) = data_batch
            if data.shape[0] == 1:
                break
            data = data.float()
            data = data.to(device)
            label = label.to(device)

            model_clf_optimizer.zero_grad()
            
            # with amp.autocast():
            # input data: (batch, 5000, 12)
            y_pred, vq_loss, perplexities = model_clf(data)
            
            recon_error = nn.BCEWithLogitsLoss(reduction="sum")(y_pred, label)
            
            loss = recon_error + vq_loss
            
            # 检查损失是否为 NaN
            if torch.isnan(loss):
                print(f"Iteration {step+1}: Loss is NaN")
                print(f"Dataloader number: {cal_num}")
                print()
                del data_batch
                del data
                del perplexities
                torch.cuda.empty_cache()
                print('clear over!')
                continue

            loss.backward()
            model_clf_optimizer.step()
            monitor.store_result(label, F.sigmoid(y_pred))
            
            # scaler.scale(loss).backward()
            # scaler.step(vqvae_optimizer)
            # scaler.step(lead_transformer_optimizer)
            # scaler.update()

            train_res_recon_error.append(recon_error.item())
            train_vq_loss.append(vq_loss.item())
            
            
            for i, perplexity in enumerate(perplexities):
                if i not in train_res_perplexity:
                    train_res_perplexity[i] = []
                train_res_perplexity[i].append(perplexity.item())
            
            # if Einthoven_loss:
            #     train_einthoven_error.append(einthoven_error.item())

            if (step+1) % 1 == 0 and cal_num % 150 == 0: # pl:5, pc: 150
                score = monitor.macro_auc_roc()
                writer.add_scalar('recon_error', np.mean(train_res_recon_error), (step+1))
                writer.add_scalar('vq_loss', np.mean(train_vq_loss), (step+1))
                writer.add_scalar('macro_auc_roc', score, (step+1))
                
                print('%d iterations' % (step+1))
                print('%d batches in train_dataloader' % cal_num)
                print('recon_error/bce loss: %.3f' % np.mean(train_res_recon_error))
                print('macro_auc_roc: %.5f' % score)
                print('vq_loss: %.3f' % np.mean(train_vq_loss))
                # if Einthoven_loss:
                #     writer.add_scalar('einthoven_loss', np.mean(train_einthoven_error), (step+1))
                #     print('einthoven_loss: %.3f' % np.mean(train_einthoven_error))

                for i, perplexity_list in train_res_perplexity.items():
                    avg_perplexity = np.mean(perplexity_list)
                    writer.add_scalar(f'perplexity_head_{i}', avg_perplexity, (step + 1))
                    print(f'perplexity (head {i}): {avg_perplexity:.3f}')
                print()
            
            # save checkpoint
            if (step+1) % 5 == 0 and cal_num % 2600 == 0 and checkpoint_path is not None:
                torch.save({
                    'epoch': step,
                    'batch': cal_num,
                    'model_clf_state_dict': model_clf.state_dict(),
                    'model_clf_optimizer_state_dict': model_clf_optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {step+1} batch {cal_num}")

        # 处理完一个epoch后，将start_batch置0
        start_batch = 0
                        
        if (step + 1) % 1 == 0:
            test_label_error = []
            monitor = Monitor()
            for (data, label) in test_dataloader:
                model_clf.eval()
                with torch.no_grad():
                    data = data.to(device)
                    label = label.to(device)
                    y_pred, _, _ = model_clf(data)
                    recon_error = nn.BCEWithLogitsLoss(reduction="sum")(y_pred, label)
                    test_label_error.append(recon_error.item())
                    monitor.store_result(label, F.sigmoid(y_pred))
            score = monitor.macro_auc_roc()
            writer.add_scalar('Test BCE loss', np.mean(test_label_error), (step+1))
            writer.add_scalar('Test macro_auc_roc', score, (step+1))
            print(f'Test BCE loss: {np.mean(test_label_error)}')
            print(f'Test macro_auc_roc: {score}')
        
        if (step + 1) % 20 == 0 and save_model_dir is not None:
            if not os.path.exists(save_model_dir):
                os.makedirs(save_model_dir)
            torch.save(model_clf, f"{save_model_dir}/clf_iter_{step+1}.pth")
            
    writer.close()
    if save_model_dir is not None:
        torch.save(model_clf, f'{save_model_dir}/clf_final.pth')


# def segmentation_vq(model_seg,
#                     train_dataloader, 
#                     test_dataloader,
#                     lr, 
#                     epoch, 
#                     device, 
#                     log_path=None, 
#                     save_model_dir = None,
#                     checkpoint_path=None):
    
#     train_res_recon_error = []
#     train_vq_loss = []
#     train_res_perplexity = {}
    
#     model_seg_optimizer = get_optimizer('adamw', model_seg, lr)
    
#     writer = SummaryWriter(log_path)
    
#     # amp initialize
#     # scaler = amp.GradScaler()
    
#     # 初始化已处理的epoch和batch数
#     start_epoch = 0
#     start_batch = 0

#     # load checkpoint with model states and epoch-batch, while using 'continue' to skip nan loss
#     if checkpoint_path is not None and os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path)
#         model_seg.load_state_dict(checkpoint['model_seg_state_dict'])
#         model_seg_optimizer.load_state_dict(checkpoint['model_seg_optimizer_state_dict'])
#         start_epoch = checkpoint['epoch']
#         start_batch = checkpoint['batch']
#         print(f"Checkpoint loaded. Resuming from epoch {start_epoch} batch {start_batch}")


#     for step in tqdm(range(start_epoch, epoch), desc="Fine-tuning Progress"):
#         # full size dataset for pre-clf
#         train_dataloader_iter = iter(train_dataloader)
#         cal_num = 0
#         # ignore the batches trained before
#         for _ in range(start_batch):
#             next(train_dataloader_iter)
#             cal_num += 1

#         for data_batch in train_dataloader_iter:
#             cal_num += 1

#             model_seg.train()
#             (data, label) = data_batch
#             # if data.shape[0] == 1:
#             #     break
#             data = data.float()
#             data = data.to(device)
#             label = label.to(device)
#             # loss = 0

#             model_seg_optimizer.zero_grad()
            
#             # with amp.autocast():
#             # input data: (batch, 12, 5000)
#             # print(data.shape)
#             y_pred, vq_loss, perplexities = model_seg(data.permute(0, 2, 1))
            
#             loss_wavep = find_wave_region(y_pred, label, 1)
#             loss_waveqrs = find_wave_region(y_pred, label, 2)
#             loss_wavet = find_wave_region(y_pred, label, 3)
#             waveloss = loss_wavep + loss_waveqrs + loss_wavet
#             loss = waveloss

#             # zero_mask = label[:,:,0] == 0  # (20, 5000)，True 表示 0
#             # # 计算第一个 0 的索引
#             # first_zero_indices = torch.where(zero_mask.any(dim=1), zero_mask.float().argmax(dim=1), torch.tensor(-1))
#             # # 计算最后一个 0 的索引
#             # last_zero_indices = torch.where(zero_mask.any(dim=1), (5000 - 1) - torch.flip(zero_mask, dims=[1]).float().argmax(dim=1), torch.tensor(-1))
            
#             # for i in range(label[:,:,0].shape[0]):
#             #     target = torch.argmax(label[i,first_zero_indices[i]:last_zero_indices[i] + 1,:], dim=-1).unsqueeze(0)
#             #     recon_error = nn.CrossEntropyLoss()(y_pred[i, :, first_zero_indices[i]:last_zero_indices[i] + 1].unsqueeze(0), target.long())
#             #     loss = loss + recon_error
#             # loss = loss + vq_loss
            
#             # # y_pred: b, 4, 5000
#             # target = torch.argmax(label[:,:,1:], dim=-1)  # 转为 (batch, 5000)
#             # recon_error = nn.CrossEntropyLoss()(y_pred, target.long())
            
#             # loss = recon_error + vq_loss
            
#             # 检查损失是否为 NaN
#             if torch.isnan(loss):
#                 print(f"Iteration {step+1}: Loss is NaN")
#                 print(f"Dataloader number: {cal_num}")
#                 print()
#                 del data_batch
#                 del data
#                 del perplexities
#                 torch.cuda.empty_cache()
#                 print('clear over!')
#                 continue

#             loss.backward()
#             model_seg_optimizer.step()
            
#             # scaler.scale(loss).backward()
#             # scaler.step(vqvae_optimizer)
#             # scaler.step(lead_transformer_optimizer)
#             # scaler.update()

#             train_res_recon_error.append(waveloss.item())
#             train_vq_loss.append(vq_loss.item())
            
            
#             for i, perplexity in enumerate(perplexities):
#                 if i not in train_res_perplexity:
#                     train_res_perplexity[i] = []
#                 train_res_perplexity[i].append(perplexity.item())

#             if (step+1) % 1 == 0 and cal_num % 10 == 0:

#                 writer.add_scalar('seg_loss', np.mean(train_res_recon_error), (step+1))
#                 writer.add_scalar('vq_loss', np.mean(train_vq_loss), (step+1))
                
#                 print('%d iterations' % (step+1))
#                 print('%d batches in train_dataloader' % cal_num)
#                 print('seg-label_loss: %.3f' % np.mean(train_res_recon_error))
#                 print('vq_loss: %.3f' % np.mean(train_vq_loss))

#                 for i, perplexity_list in train_res_perplexity.items():
#                     avg_perplexity = np.mean(perplexity_list)
#                     writer.add_scalar(f'perplexity_head_{i}', avg_perplexity, (step + 1))
#                     print(f'perplexity (head {i}): {avg_perplexity:.3f}')
#                 print()
            
#             # save checkpoint
#             if (step+1) % 100 == 0 and cal_num % 2400 == 0 and checkpoint_path is not None:
#                 torch.save({
#                     'epoch': step,
#                     'batch': cal_num,
#                     'model_seg_state_dict': model_seg.state_dict(),
#                     'model_seg_optimizer_state_dict': model_seg_optimizer.state_dict(),
#                 }, checkpoint_path)
#                 print(f"Checkpoint saved at epoch {step+1} batch {cal_num}")

#         # 处理完一个epoch后，将start_batch置0
#         start_batch = 0
                        
#         if (step + 1) % 1 == 0:
#             test_label_error = []
#             # monitor = Monitor()
#             for (data, label) in test_dataloader:
#                 model_seg.eval()
#                 with torch.no_grad():
#                     data = data.to(device)
#                     label = label.to(device)
#                     loss = 0
#                     y_pred, vq_loss, perplexities = model_seg(data.permute(0, 2, 1))
#                     loss_wavep = find_wave_region(y_pred, label, 1)
#                     loss_waveqrs = find_wave_region(y_pred, label, 2)
#                     loss_wavet = find_wave_region(y_pred, label, 3)
#                     waveloss = loss_wavep + loss_waveqrs + loss_wavet
#                     # loss = waveloss + vq_loss

#                     # zero_mask = label[:,:,0] == 0  # (20, 5000)，True 表示 0
#                     # # 计算第一个 0 的索引
#                     # first_zero_indices = torch.where(zero_mask.any(dim=1), zero_mask.float().argmax(dim=1), torch.tensor(-1))
#                     # # 计算最后一个 0 的索引
#                     # last_zero_indices = torch.where(zero_mask.any(dim=1), (5000 - 1) - torch.flip(zero_mask, dims=[1]).float().argmax(dim=1), torch.tensor(-1))
#                     # for i in range(label[:,:,0].shape[0]):
#                     #     target = torch.argmax(label[i,first_zero_indices[i]:last_zero_indices[i] + 1,:], dim=-1).unsqueeze(0)
#                     #     recon_error = nn.CrossEntropyLoss()(y_pred[i, :, first_zero_indices[i]:last_zero_indices[i] + 1].unsqueeze(0), target.long())
#                     #     loss = loss + recon_error
                    
#                     test_label_error.append(waveloss.item())
#             writer.add_scalar('Test loss', np.mean(test_label_error), (step+1))
#             print(f'Test loss: {np.mean(test_label_error)}')
        
#         if (step + 1) % 25 == 0 and save_model_dir is not None:
#             if not os.path.exists(save_model_dir):
#                 os.makedirs(save_model_dir)
#             torch.save(model_seg, f"{save_model_dir}/seg_iter_{step+1}.pth")
            
#     writer.close()
#     if save_model_dir is not None:
#         torch.save(model_seg, f'{save_model_dir}/seg_final.pth')
    

if __name__ == '__main__':
    print('The whole model has been loaded!')



