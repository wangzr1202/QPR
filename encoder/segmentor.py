import numpy as np
import torch
import torch.nn as nn
import math
from torch_ecg.utils.utils_nn import adjust_cnn_filter_lengths
from torch_ecg.model_configs import ECG_UNET_VANILLA_CONFIG
# from torch_ecg.models.ecg_crnn import ECG_CRNN
from torch_ecg.models.unets.ecg_unet import ECG_UNET


class model_seg(nn.Module):
    def __init__(self, model_vqvae):
        super(model_seg, self).__init__()
        self.model_vqvae = model_vqvae
        config = adjust_cnn_filter_lengths(ECG_UNET_VANILLA_CONFIG, fs=500)
        classes = ["None", "P", "QRS", "T"]
        self.segmentor = ECG_UNET(classes, 12, config)
        # self.segmentor = S4(d_input=12, d_model=4)

    def forward(self, data):
        # data: 5000, 12
        z = self.model_vqvae._encoder(data) # z keys: 'last_hidden_state', 'loc', 'scale', 'patch_input'

        # z.last_hidden_state: (batch, 12, patches, patch_length)
        hidden_state = self.model_vqvae.BN(z.last_hidden_state)

        quantization, indices, loss = self.model_vqvae._vq(hidden_state) # (batch, 12, 625/500, 128)

        quantized_de = self.model_vqvae._map(quantization).reshape(data.shape[0], 12, -1) # batch, 12, 5000
        if quantized_de.shape[-1] < 5000:
            quantized_de = torch.cat((quantized_de, quantized_de[:, :, -8:]), dim=-1)
        quantized_de = self.model_vqvae.BN1d(quantized_de) # 12, 5000
        
        y_pred = self.segmentor(quantized_de) # y_pred size: (batch, 5000, 4)
        
        vq_loss = loss.sum() / self.model_vqvae.vq_heads
        perplexities = self.model_vqvae.perplexity_cal(indices, self.model_vqvae.num_embeddings, self.model_vqvae.vq_heads)

        return y_pred.permute(0, 2, 1), vq_loss, perplexities
    

# modified!!
def process_labels(labels):
    # 转换为numpy数组
    labels = np.asarray(labels.cpu())
    batch_size, seq_len = labels.shape
    
    left = np.zeros_like(labels)
    left[:, 1:] = labels[:, :-1]
    left[:, 0] = 1  # 左边界左侧视为非0
    
    right = np.zeros_like(labels)
    right[:, :-1] = labels[:, 1:]
    right[:, -1] = 1  # 右边界右侧视为非0
    
    transition_mask = (labels == 1) & ((left == 0) | (right == 0))
    transitions = [np.where(row)[0] for row in transition_mask]
    transitions = [t for t in transitions if t.size > 0]

    truncated = []
    for i in range(batch_size):
        # 获取当前样本的1的位置
        ones_indices = np.where(transition_mask[i])[0]
        
        if len(ones_indices) == 0:
            # 无过渡点时保留空数组
            pass
            # truncated.append(np.array([], dtype=int))
        else:
            # 计算截取范围
            start = ones_indices.min()
            end = ones_indices.max()
            # 截取区间包含端点
            truncated.append(labels[i, start:end+1])
    return transitions, truncated

# modified!!
def find_wave_region(y_pred, label, wave_index):    
    loss = []
    transitions, truncated_label = process_labels(label[:,:,wave_index])
    for i in range(len(truncated_label)):
        target = torch.tensor(truncated_label[i]).unsqueeze(0).to('cuda')
        prediction = y_pred[i, wave_index, transitions[i][0]:transitions[i][-1]+1].unsqueeze(0)
        recon_error = nn.CrossEntropyLoss()(prediction, target) / target.shape[-1]
        loss.append(recon_error)
    return sum(loss) / len(loss)

# def find_wave_region(y_pred, label, wave_index):
#     zero_mask = label[:,:,wave_index] == 1
#     valid_rows = zero_mask.any(dim=1)
#     zero_mask = zero_mask[valid_rows]
#     # 计算第一个 1 的索引
#     first_indices = torch.where(zero_mask.any(dim=1), zero_mask.float().argmax(dim=1), torch.tensor(-1))
#     # 计算最后一个 1 的索引
#     last_indices = torch.where(zero_mask.any(dim=1), (5000 - 1) - torch.flip(zero_mask, dims=[1]).float().argmax(dim=1), torch.tensor(-1))
#     loss = []
#     for i in range(zero_mask.shape[0]):
#         target = label[i , first_indices[i]:last_indices[i]+1 , wave_index].unsqueeze(0) # 1, length
#         prediction = y_pred[i , wave_index , first_indices[i]:last_indices[i]+1].unsqueeze(0) # 1, length
#         recon_error = nn.CrossEntropyLoss()(prediction, target) / target.shape[-1]
#         loss.append(recon_error)
#     return sum(loss) / len(loss)