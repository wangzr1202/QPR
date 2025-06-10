### data preprocessing functions and some tool functions

import yaml
import torch
import numpy as np
import random
import copy
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib as mpl

# json config load
class Config:
    def __init__(self, config_file):
        with open(config_file, "r") as f:
            # self._config = json.load(f)
            self._config = yaml.safe_load(f)

    def __getattr__(self, name):
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")


class Subsample(object):
    """
    Subsample fixed length of ECG signals.

    Args:
        subsample_length (int): Length of subsampled data.
    """
    def __init__(self, subsample_length: int):

        assert isinstance(subsample_length, int)
        self.subsample_length = subsample_length

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (12, sequence_length).,
                            "label": label}
        Returns:
            sample (Dict): {"data": Array of shape (12, subsample_length).,
                            "label": label}
        """
        data, label = sample["data"], sample["label"]

        start = np.random.randint(0, data.shape[1] - self.subsample_length)
        subsampled_data = data[:, start:start+self.subsample_length]

        return {"data": subsampled_data, "label": label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    # def __init__(self, label_type: str="float"):
    #     self.label_type = label_type

    def __call__(self, sample):
        data, label  = sample["data"], sample["label"]
        data_tensor = torch.from_numpy(data)
        label_tensor = torch.from_numpy(label)
        # data_tensor = data_tensor.float()
        # if self.label_type == "float":
        #     label_tensor = label_tensor.float()
        # elif self.label_type == "long":
        #     label_tensor = label_tensor.long()
        # else:
        #     raise NotImplementedError
        return {"data": data_tensor, "label": label_tensor}

class SubsampleEval(Subsample):
    """
    Subsampling for evaluation mode.

    Args:
        subsample_length (int): Length of subsampled data.
    """

    def _pad_signal(self, data):
        """
        Args:
            data (np.ndarray):
        Returns:
            padded_data (np.ndarray):
        """
        chunk_length = self.subsample_length // 2
        pad_length = chunk_length - data.shape[1] % chunk_length

        if pad_length == 0:
            return data
        pad = np.zeros([12, pad_length])
        pad_data = np.concatenate([data, pad], axis=-1)
        return pad_data

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (12, sequence_length).,
                            "label": label}
        Returns:
            sample (Dict): {"data": Array of shape (12, num_split, subsample_length).,
                            "label": label}
        """
        data, label = sample["data"], sample["label"]
        slice_indices = np.arange(0, data.shape[1], self.subsample_length // 2)
        index_range = np.arange(self.subsample_length)
        target_locs = slice_indices[:, np.newaxis] + index_range[np.newaxis]

        padded_data = self._pad_signal(data)
        try:
            eval_subsamples = padded_data[:, target_locs]
        except:
            eval_subsamples = padded_data[:, target_locs[:-1]]
        return {"data": eval_subsamples, "label": label}

class ProcessLabel(object):
    "Convert to multiclass label"

    def __init__(
        self,
        normal_index: int,
        target_index: int,
        num_classes: int = 3
    ) -> None:
        self.normal_index = normal_index
        self.target_index = target_index
        self.num_classes = num_classes

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (12, sequence_length).,
                            "label": label}
        Returns:
            sample (Dict): {"data": Array of shape (12, num_split, subsample_length).,
                            "label": label}
        """
        data, label = sample["data"], sample["label"]

        if label[self.target_index]:
            processed_label = 1
        elif label[self.normal_index]:
            processed_label = 0
        else:
            processed_label = 2
        processed_label = np.array(processed_label)

        return {"data": data, "label": processed_label}


def _calc_class_weight(labels, normal_index, target_index):
    """
    Calculate class weight for target dx and others (1 for normal labels).

    Args:
        labels (np.ndarray): Label data array of shape [num_sample, num_classes]
    Returns:
        class_weight (np.ndarray):
    """
    num_samples = labels.shape[0]
    # Extract normal and target dx labels
    normal_labels = labels[:, normal_index]
    target_labels = labels[:, target_index]
    # Validate normal and target dx label do not overlap
    # assert((normal_labels & target_labels).sum() == 0)

    num_normal = normal_labels.sum()
    num_target = target_labels.sum()
    num_others = num_samples - (num_normal + num_target)

    class_weights = [1, num_target/num_normal, num_others/num_normal]
    return np.array(class_weights)

# augmentation
def add_noise(x, noise_level = 0.01):
    noise = torch.randn_like(x) * noise_level
    noisy_data = x + noise
    # noisy_data = torch.clip(noisy_data, 0. ,1.)
    return noisy_data


# origin data preprocessing
# input data npy: 78w with only 0-1 normalization
# due to the large size of the dataset, we divided the npy into 6 pieces
# data is a dataset, keys: data,
def remove_nan_or_inf(dataset):
    """
    check NaN or Inf
    """
    invalid_samples = np.any(np.isnan(dataset.data) | np.isinf(dataset.data), axis=(1, 2))
    dataset.data = dataset.data[~invalid_samples]
    dataset.labels = dataset.labels[~invalid_samples]
    return dataset

def remove_nan_or_inf_inplace(dataset):
    invalid_mask = np.logical_or(np.isnan(dataset.data), np.isinf(dataset.data))
    invalid_samples = np.any(invalid_mask, axis=(1, 2))
    dataset.data = dataset.data[~invalid_samples]
    dataset.labels = dataset.labels[~invalid_samples]
    return dataset

def remove_out_of_range(dataset, min_val=-11, max_val=11):
    """
    check out-of-range data
    """
    mask = (dataset.data < min_val) | (dataset.data > max_val)
    invalid_samples = np.any(mask, axis=(1, 2))
    dataset.data = dataset.data[~invalid_samples]
    dataset.labels = dataset.labels[~invalid_samples]
    return dataset

def remove_statistical_outliers(dataset, mean_range=(-1.5, 1.5), std_range=(0.05, 2)):
    """
    check the mean and std
    """
    means = np.mean(dataset.data, axis=1)
    stds = np.std(dataset.data, axis=1)

    invalid_samples = (
        (means < mean_range[0]) | (means > mean_range[1]) | (stds < std_range[0]) | (stds > std_range[1])
    ).any(axis=1)
    dataset.data = dataset.data[~invalid_samples]
    dataset.labels = dataset.labels[~invalid_samples]
    return dataset

# def remove_spikes_batch(data, diff_threshold=10, batch_size=1000):
#     """
#     check spike in batch
#     """
#     clean_data = []
#     for i in range(0, data.shape[0], batch_size):
#         batch = data[i:i + batch_size]
#         diffs = np.diff(batch, axis=1)
#         spikes = np.abs(diffs) > diff_threshold
#         invalid_samples = np.any(spikes, axis=(1, 2))
#         clean_batch = batch[~invalid_samples]
#         clean_data.append(clean_batch)
#     return np.concatenate(clean_data, axis=0)

def remove_spikes(dataset, diff_threshold=10):
    """
    check spike
    """
    diffs = np.diff(dataset.data, axis=1)  # 计算相邻点的差分
    spikes = np.abs(diffs) > diff_threshold  # 标记跳变
    invalid_samples = np.any(spikes, axis=(1, 2))  # 标记有跳变的样本
    dataset.data = dataset.data[~invalid_samples]
    dataset.labels = dataset.labels[~invalid_samples]
    return dataset

def clean_ecg_data(dataset, min_val=-11, max_val=11, mean_range=(-1.5, 1.5), std_range=(0.05, 2), diff_threshold=10):
    dataset = remove_nan_or_inf_inplace(dataset)
    dataset = remove_out_of_range(dataset, min_val=-11, max_val=11)
    dataset = remove_statistical_outliers(dataset, mean_range=(-1.5, 1.5), std_range=(0.05, 2))
    dataset = remove_spikes(dataset, diff_threshold=10)
    return dataset

# def clean_ecg_data_in_batches(data, batch_size=2000):
#     cleaned_batches = []
#     for i in range(0, data.shape[0], batch_size):
#         batch = data[i:i + batch_size]
#         batch = clean_ecg_data(batch)
#         cleaned_batches.append(batch)
#     return np.concatenate(cleaned_batches, axis=0)

def select_dataset(dataset, index):
    # dataset.data:17443, 5000,12, dataset.labels:17443, 71
    # NORM: 4
    # STE: 57, totally 28 samples
    # LVH: 7, totally 2137 samples
    dataset_new = copy.deepcopy(dataset)
    selected_labels = dataset_new.labels[:, index] == 1.
    # 用一个新的dataset来存储
    dataset_new.data = dataset_new.data[selected_labels]
    dataset_new.labels = dataset_new.labels[selected_labels]
    return dataset_new


def make_sample_mask_with_overlap_false(patch_mask, 
                                        signal_length=5000, 
                                        patch_size=16, 
                                        stride=8):
    """
    patch_mask: (1, 12, 624) 布尔矩阵
    signal_length: 信号长度（如 5000）
    patch_size: 每个 patch 的长度，默认为 16
    stride:     patch 的步长，默认为 8

    规则：
    1) 若 patch_mask[0, ch, i] == False，则整段 [i*8, i*8+16) 标记为 False。
    2) 若 patch i 和 patch i+1 有一个为 False，则它们重叠部分 
       [i*8+(patch_size - stride), i*8+patch_size) 标记为 False
       即 [i*8+8, i*8+16) 大小=8 的区间。
    """

    batch, num_ch, num_patches = patch_mask.shape
    assert batch == 1, "本示例假设 batch=1"
    sample_mask = np.ones((1, signal_length, num_ch), dtype=bool)

    for ch in range(num_ch):
        # 步骤 1: 若当前 patch 为 False，则其覆盖的 16 个采样都置 False
        for i in range(num_patches):
            if not patch_mask[0, ch, i]:
                start = i * stride
                end = start + patch_size
                sample_mask[0, start:end, ch] = False

        # 步骤 2: 对相邻 patch 间的重叠部分做额外处理
        # 只要 patch i 或 patch i+1 为 False，就把重叠区间标记为 False
        for i in range(num_patches - 1):
            if (not patch_mask[0, ch, i]) or (not patch_mask[0, ch, i+1]):
                overlap_start = i*stride + (patch_size - stride)  # i*8 + 8
                overlap_end   = i*stride + patch_size              # i*8 + 16
                sample_mask[0, overlap_start:overlap_end, ch] = False
    
    return sample_mask
    
'''
# original
def plot_ecg_with_full_mask(data, data_recon, sample_mask, save_path, num_leads = 1, time_len = 2500, start_point = 100, save_model_name = 'PatchTST'):
    """
    data: shape (1, 5000, 12) 的 ECG 信号
    sample_mask: shape (1, 5000, 12)，逐点布尔掩码（True 表示该采样被 mask）
    """
    assert data.shape[0] == 1, "test 1 st sample in test_dataloader"
    assert data.shape == sample_mask.shape, "data 与 sample_mask 形状应一致"
    assert data.shape == data_recon.shape, "data 与 data_recon 形状应一致"

    fig, axes = plt.subplots(num_leads, 1, figsize=(9, 3*num_leads), sharex=True)

    # 如果只有一个通道，axes 不是列表，需要单独处理
    if num_leads == 1:
        axes = [axes]

    for ch in range(num_leads):
        ax = axes[ch]

        # 取出该导联信号和 mask
        ecg_signal = data[0, start_point:time_len, ch]
        ecg_signal_recon = data_recon[0, start_point:time_len, ch]
        mask_for_ch = sample_mask[0, start_point:time_len, ch]  # shape (5000,)
        first_plot = True

        # 画原始 ECG 波形
        x = np.arange(time_len-start_point)
        # ax.plot(x, ecg_signal, label=f'Lead {ch + 1}')
        ax.plot(x, ecg_signal, label = 'Raw signal')

        # 找到 mask==True 的连续区间做可视化标记
        start_idx = None
        for i in range(time_len - start_point):
            if mask_for_ch[i] and start_idx is None:
                start_idx = i
            elif not mask_for_ch[i] and start_idx is not None:
                ax.axvspan(start_idx, i, color='#929591', alpha=0.2)
                
                # 仅在第一次绘制时添加 label
                if first_plot:
                    ax.plot(x[start_idx:i], ecg_signal_recon[start_idx:i], 
                            color='C1', linewidth=0.75, label=save_model_name)
                    first_plot = False  # 标记已添加过 label
                else:
                    ax.plot(x[start_idx:i], ecg_signal_recon[start_idx:i], 
                            color='C1', linewidth=0.75)  # 不带 label 的绘制

                start_idx = None
        # 若最后还处于被 mask 区间，补画收尾
        if start_idx is not None:
            ax.axvspan(start_idx, time_len-start_point, color='#929591', alpha=0.2)

        ax.set_ylabel("value")
        ax.legend(loc='upper right')

    # plt.xlabel("Time")
    # plt.suptitle("Lead I with Mask and Resconstruction(0~5s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi = 400)
    plt.show()
'''


# def plot_ecg_with_full_mask(data, data_recons, sample_mask, save_path, num_leads = 1, time_len = 2500, start_point = 100, save_model_name = 'PatchTST'):
#     """
#     data: shape (1, 5000, 12) 的 ECG 信号
#     sample_mask: shape (1, 5000, 12)，逐点布尔掩码（True 表示该采样被 mask）
#     """
#     mpl.rcParams['font.family'] = 'sans-serif'
#     mpl.rcParams['font.size'] = 5
#     assert data.shape[0] == 1, "test 1 st sample in test_dataloader"
#     assert data.shape == sample_mask.shape, "data 与 sample_mask 形状应一致"
#     # assert data.shape == data_recons[0].shape, "data_recon 为多个随机种子的数组，大小为5，5000，12"

#     # min_data_recon = np.min(data_recons, axis=0)
#     # max_data_recon = np.max(data_recons, axis=0)
#     # mean_data_recon = np.mean(data_recons, axis=0) 
#     # data_recon_95 = np.percentile(data_recons, 97.5, axis=0)
#     # data_recon_5 = np.percentile(data_recons, 2.5, axis=0)   
    
#     fig, axes = plt.subplots(num_leads, 1, figsize=(55/25.4, 18/25.4), sharex=True)

#     # 如果只有一个通道，axes 不是列表，需要单独处理
#     if num_leads == 1:
#         axes = [axes]

#     for ch in range(num_leads):
#         ax = axes[ch]

#         # 取出该导联信号和 mask
#         ecg_signal = data[0, start_point:time_len, ch]
#         ecg_signal_recon = data_recons[0, start_point:time_len, ch]
#         mask_for_ch = sample_mask[0, start_point:time_len, ch]  # shape (5000,)
#         first_plot = True
        
#         # # 参数设置
#         # fs = 500  # 采样率（Hz）
#         # cutoff_freq = 30  # 截止频率（Hz）
#         # nyquist = 0.5 * fs
#         # order = 128  # FIR滤波器阶数

#         # # 设计FIR低通滤波器（窗函数法）
#         # taps = signal.firwin(
#         #     numtaps=order,
#         #     cutoff=cutoff_freq,
#         #     fs=fs,
#         #     window='hamming'  # 可选：'blackman', 'hann'
#         # )

#         # def apply_filter(signal_data):
#         #     filtered = signal.filtfilt(taps, [1.0], signal_data)
#         #     return filtered
        
#         # ecg_signal = apply_filter(ecg_signal)
#         # ecg_signal_recon = apply_filter(ecg_signal_recon)
#         # print(ecg_signal.shape, ecg_signal_recon.shape)

#         # 画原始 ECG 波形
#         x = np.arange(time_len-start_point)
#         # ax.plot(x, ecg_signal, label=f'Lead {ch + 1}')
#         ax.plot(x, ecg_signal, linewidth=0.6)

#         # 找到 mask==True 的连续区间做可视化标记
#         start_idx = None
#         for i in range(time_len - start_point):
#             if mask_for_ch[i] and start_idx is None:
#                 start_idx = i
#             elif not mask_for_ch[i] and start_idx is not None:
#                 ax.axvspan(start_idx, i, color='#929591', alpha=0.2)
#                 ax.fill_between(x[start_idx:i], data_recons[start_idx+start_point:i+start_point, ch], data_recons[start_idx+start_point:i+start_point, ch], color='green', alpha=0.2)
#                 # 仅在第一次绘制时添加 label
#                 if first_plot:
#                     ax.plot(x[start_idx:i], ecg_signal_recon[start_idx:i], 
#                             color='C1', linewidth=0.6)
#                     first_plot = False  # 标记已添加过 label
#                 else:
#                     ax.plot(x[start_idx:i], ecg_signal_recon[start_idx:i], 
#                             color='C1', linewidth=0.6)  # 不带 label 的绘制

#                 start_idx = None
#         # 若最后还处于被 mask 区间，补画收尾
#         if start_idx is not None:
#             ax.axvspan(start_idx, time_len-start_point, color='#929591', alpha=0.2)

#         # ax.set_ylabel("value", fontsize=6)
#         # ax.tick_params(axis='both', labelsize=6)  # 刻度字体大小
#         # ax.tick_params(axis='both', length=0)  # length=0 隐藏刻度线
#         # ax.set_xticks(np.arange(0, 751, 250))  # x 轴从 0 到 10，间隔为 2
#         # ax.set_xticklabels([])
#         # ax.set_yticklabels([])
#         # ax.legend(loc='lower right')

#     # plt.xlabel("Time")
#     # plt.suptitle("Lead I with Mask and Resconstruction(0~5s)")
#     # plt.tight_layout() # 自动调整间距
#     plt.ylabel('value', fontsize=mpl.rcParams['font.size'])
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#     plt.grid(True)
#     plt.savefig(save_path, dpi = 600, transparent=True, pad_inches=0, bbox_inches='tight')
#     plt.show()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Einthoven leads transformation
# 12 leads: I, II, III, aVR, aVL, aVF, V1~V6
T_einthoven_12 = torch.tensor([[1, 0, -1, -0.5, 1, -0.5],
                               [0, 1, 1, -0.5, -0.5, 1],
                               [0, 0, 0, 0, 0, 0]]) # 以12导联为标准
T_einthoven_13 = torch.tensor([[1, 1, 0, -1, 0.5 , 0.5],
                               [0, 0, 0, 0, 0, 0],
                               [0, 1, 1, -0.5, -0.5, 1]]) # 以13导联为标准
T_einthoven_23 = torch.tensor([[0, 0, 0, 0, 0, 0],
                               [1, 1, 0, -1, 0.5, 0.5],
                               [-1, 0, 1, 0.5, -1, 0.5]]) # 以23导联为标准
T_identity = torch.eye(6)

adder_mean = (T_einthoven_12 + T_einthoven_13 + T_einthoven_23)/3

adder = torch.cat((sum(adder_mean), 0.5 * torch.ones(6)), dim = -1) # dim = 12

T_einthoven_matrices = [T_einthoven_12, T_einthoven_13, T_einthoven_23]
T_einthoven_all = []

for T in T_einthoven_matrices:
    T_12lead = torch.zeros(12, 12)
    T_12lead[:3, :6] = T  # 填入前3行和前6列
    T_12lead[6:, 6:] = T_identity  # 填入后6x6单位阵
    T_einthoven_all.append(T_12lead)

T_einthoven_12all, T_einthoven_13all, T_einthoven_23all = T_einthoven_all
