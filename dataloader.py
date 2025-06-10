'''
For preprocessing, we refer some of the methods in the following paper and codes.
paper: In-depth Benchmarking of Deep Neural Network  Architectures for ECG Diagnosis;
github: https://github.com/seitalab/dnn_ecg_comparison
'''

import random
import numpy as np
from typing import Type
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, Subset
import utils.utils as utils


class ptbxldataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path, allow_pickle=True) 
        self.labels = np.load(label_path, allow_pickle=True)
        self.data = self.data.astype(np.float32)
        self.labels = self.labels.astype(np.float32)
        assert self.data.shape[0] == self.labels.shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = torch.tensor(self.data[idx], dtype=torch.float32)
        sample_label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample_data, sample_label

class g12ecdataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path, allow_pickle=True) 
        self.labels = np.load(label_path, allow_pickle=True)
        self.data = self.data.astype(np.float32)
        self.labels = self.labels.astype(np.float32)
        assert self.data.shape[0] == self.labels.shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = torch.tensor(self.data[idx], dtype=torch.float32)
        sample_label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample_data, sample_label
    
class ludbdataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path, allow_pickle=True) 
        self.labels = np.load(label_path, allow_pickle=True)
        self.data = self.data.astype(np.float32)
        self.labels = self.labels.astype(np.float32)
        assert self.data.shape[0] == self.labels.shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = torch.tensor(self.data[idx], dtype=torch.float32)
        sample_label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample_data, sample_label


class mimic_ecgdataset(Dataset):
    def __init__(self, data_path, data_size):
        self.data = np.memmap(data_path, dtype='float32', mode='r', shape = (data_size, 5000, 12))
        self.data = np.array(self.data)
        # self.data = np.load(data_path, allow_pickle = True)
        # self.data = self.data.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample_data = torch.tensor(self.data[idx], dtype=torch.float32)
        return sample_data

# MC = multi-class, we select the chosen labels from the original dataset
class ptbxldataset_MC(Dataset):
    def __init__(self, data_path, label_path, transform):
        self.data = np.load(data_path, allow_pickle=True) 
        self.labels = np.load(label_path, allow_pickle=True)

        self.data = self.data.astype(np.float32)
        self.labels = self.labels.astype(np.float32)

        self.data = np.transpose(self.data, (0, 2, 1))
        self.transform = transform
        assert self.data.shape[0] == self.labels.shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels = self.data[index], self.labels[index]

        sample = {"data": data, "label": labels}
        sample = self.transform(sample)
        sample_data, sample_label = sample["data"], sample["label"]

        # sample_data = torch.tensor(sample["data"], dtype=torch.float32)
        # sample_label = torch.tensor(sample["label"], dtype=torch.float32)
        return sample_data, sample_label
    
class g12ecdataset_MC(Dataset):
    def __init__(self, data_path, label_path, transform):
        self.data = np.load(data_path, allow_pickle=True) 
        self.labels = np.load(label_path, allow_pickle=True)

        self.data = self.data.astype(np.float32)
        self.labels = self.labels.astype(np.float32)

        self.data = np.transpose(self.data, (0, 2, 1))
        self.transform = transform
        assert self.data.shape[0] == self.labels.shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels = self.data[index], self.labels[index]

        sample = {"data": data, "label": labels}
        sample = self.transform(sample)
        sample_data, sample_label = sample["data"], sample["label"]

        # sample_data = torch.tensor(sample["data"], dtype=torch.float32)
        # sample_label = torch.tensor(sample["label"], dtype=torch.float32)
        return sample_data, sample_label


def prepare_dataloader_multiclass(config,
                                  task_name,
                                  dataset_name,
                                  batch_size,
                                  frequency,
                                  length) -> Type[DataLoader]:

    print('Preparing dataloader for multi-classification...')
    normal_index = config.MULTICLASS_LABELS_INDEX["Normal"][dataset_name]
    target_index = config.MULTICLASS_LABELS_INDEX[task_name][dataset_name]

    transformations = prepare_preprocess_multiclass(
        frequency, length, normal_index, target_index, True)
    
    transformations_test = prepare_preprocess_multiclass(
        frequency, length, normal_index, target_index, False)

    if dataset_name == "ptbxl-500":
        root_a = [config.root_ptbxl500 + "all" + config.folder_group[0]] * 6
        root_b = config.npy_files
        ptbxl500_path = [a + b for a, b in zip(root_a, root_b)]
        # train_data_path, train_labels_path, test_data_path, test_labels_path, val_data_path, val_labels_path = config.ptbxl500_path
        (train_data_path, train_labels_path,
        test_data_path, test_labels_path, 
        val_data_path, val_labels_path) = ptbxl500_path
        dataset_train = ptbxldataset_MC(train_data_path, train_labels_path, transformations)
        dataset_valid = ptbxldataset_MC(val_data_path, val_labels_path, transformations)
        dataset_test = ptbxldataset_MC(test_data_path, test_labels_path, transformations_test)

    elif dataset_name == "g12ec":
        root_a = [config.root_g12ec + config.folder_group[0]] * 6
        root_b = config.npy_files
        g12ec_path = [a + b for a, b in zip(root_a, root_b)]
        (train_data_path, train_labels_path,
        test_data_path, test_labels_path, 
        val_data_path, val_labels_path) = g12ec_path
        dataset_train = g12ecdataset_MC(train_data_path, train_labels_path, transformations)
        dataset_valid = g12ecdataset_MC(val_data_path, val_labels_path, transformations)
        dataset_test = g12ecdataset_MC(test_data_path, test_labels_path, transformations_test)

    dataset_train_concat = ConcatDataset([dataset_train, dataset_valid])
    train_dataloader = DataLoader(dataset_train_concat, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    labels = dataset_train.labels
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
    weight = np.array(class_weights)
    return train_dataloader, test_dataloader, weight


def prepare_preprocess_multiclass(
    frequency: int,
    length: int,
    normal_index: int,
    target_index: int,
    is_train: bool
) -> Type[transforms.Compose]:
    """
    Prepare and compose transform functions.
    Args:
        frequency (int):
        length (int):
        target_index (int):
        normal_index (int):
        is_train (bool):
    Returns:
        composed
    """
    subsample_length = int(frequency * length)
    if is_train:
        composed = transforms.Compose([
            # utils.Subsample(subsample_length),
            utils.ProcessLabel(normal_index, target_index),
            utils.ToTensor()
        ])
    else:
        composed = transforms.Compose([
            # utils.SubsampleEval(subsample_length),
            utils.ProcessLabel(normal_index, target_index),
            utils.ToTensor()
        ])
    return composed



def create_few_shot_unique_label_dataset(original_dataset, 
                                         num_samples_per_labelset=1, 
                                         seed=2025):
    """
    将 71 维多标签向量视为一个“标签组合”，若两个样本标签向量不同，
    则视为不同类别（labelset）。在每个标签组合下随机抽取指定数量的样本。

    original_dataset: 其 __getitem__ 返回 (data, label)，其中 label.shape=(71,)，可能有多个 1
    num_samples_per_labelset: 每个标签组合随机抽取多少条样本
    """
    # random.seed(seed)

    # 1) 根据完整的标签向量将索引分组
    #    由于 label 是 71 维的 float32 Tensor，可转为 tuple 作为字典键
    labelset_to_indices = {}
    for idx in range(len(original_dataset)):
        _, label = original_dataset[idx]    # label: shape [71]
        # 将标签向量转换成 tuple 或其他可哈希表示，确保同样本标签可归一
        # 注意要把 label 转成 int 或 bool，以避免浮点比较造成错误
        label_np = label.numpy().astype(int)  # [71,]
        label_tuple = tuple(label_np.tolist())
        
        if label_tuple not in labelset_to_indices:
            labelset_to_indices[label_tuple] = []
        labelset_to_indices[label_tuple].append(idx)

    # 2) 在每个“标签组合”里随机抽取指定数量
    selected_indices = []
    for label_tuple, indices in labelset_to_indices.items():
        if len(indices) < num_samples_per_labelset:
            # 当此标签组合下样本数不足时，可选择跳过或特殊处理
            continue
        sampled = random.sample(indices, num_samples_per_labelset)
        selected_indices.extend(sampled)

    # 3) 用这些索引构造新的子集
    few_shot_subset = Subset(original_dataset, selected_indices)
    return few_shot_subset


def create_few_shot_multilabel_dataset(original_dataset, 
                                       num_samples_per_class=1, 
                                       seed=2025,
                                       allow_duplicate=False):
    """
    针对多标签 (如 71 维 one-hot) 的数据集，为每个标签随机抽取 num_samples_per_class 个样本。
    若某个样本同时属于多个标签，则它会被添加到多个标签的列表中 (可能导致重复)。
    allow_duplicate 表示是否允许重复出现在最终结果中。
    """
    # random.seed(seed)

    # 1) 根据标签将索引分组（一个样本含多标签时，归入多个标签组）
    class_to_indices = {}
    for idx in range(len(original_dataset)):
        data, label = original_dataset[idx]  # label.shape = [71], 多热向量
        # 找出 label 中为 1 的位置 (既是该样本所属的所有类别)
        # 若您判定阈值不同，也可自行修改判定条件
        label_positions = (label >= 0.5).nonzero(as_tuple=True)[0]  # tensor([...])
        
        for class_id_t in label_positions:
            class_id = int(class_id_t.item())
            if class_id not in class_to_indices:
                class_to_indices[class_id] = []
            class_to_indices[class_id].append(idx)

    # 2) 在每个类别的 index 列表中随机抽取指定数量的索引
    selected_indices = []
    for class_id, indices in class_to_indices.items():
        if len(indices) < num_samples_per_class:
            # 如果某标签对应的样本数不足，可选择跳过或其他处理
            continue
        sampled = random.sample(indices, num_samples_per_class)
        selected_indices.extend(sampled)

    # 如果不允许同一个样本重复出现多次 (当其属于多个标签)，
    # 可以在这里去重
    if not allow_duplicate:
        selected_indices = list(set(selected_indices))

    # 3) 用这些索引创建 Subset
    few_shot_subset = Subset(original_dataset, selected_indices)
    return few_shot_subset


def get_dataloader(mode='pretrain', config=None, dataset_name=None, batch_size=None, shuffle=True):
    '''
    不需要指定路径, 只需要指定mode、dataset_name、batch_size;
    不支持100Hz
    '''
    if mode == 'finetune':
        if dataset_name is None:
            raise ValueError("Dataset name must be provided for pretrain mode.")
        
        dataset_map = {
            'ptbxl-500': ptbxldataset,
            'ptbxl-100': ptbxldataset,
            'ptbxl-few-shot': ptbxldataset,
            'g12ec': g12ecdataset,
            'ludb': ludbdataset,
        }
                
        if dataset_name == 'ptbxl-500':
            print('Preparing dataloader for ptbxl-500...')
            root_a = [config.root_ptbxl500 + config.TASKS_ptbxl[0] + config.folder_group[0]] * 6
            root_b = config.npy_files
            ptbxl500_path = [a + b for a, b in zip(root_a, root_b)]
            # train_data_path, train_labels_path, test_data_path, test_labels_path, val_data_path, val_labels_path = config.ptbxl500_path
            (train_data_path, train_labels_path,
            test_data_path, test_labels_path, 
            val_data_path, val_labels_path) = ptbxl500_path

            dataset_train = dataset_map[dataset_name](train_data_path, train_labels_path)
            dataset_test = dataset_map[dataset_name](test_data_path, test_labels_path)
            dataset_valid = dataset_map[dataset_name](val_data_path, val_labels_path)
            # dataset_train = utils.clean_ecg_data(dataset_train)
            # dataset_test = utils.clean_ecg_data(dataset_test)
            # dataset_valid = utils.clean_ecg_data(dataset_valid)
            dataset_train = ConcatDataset([dataset_train, dataset_valid])
            
            train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
            test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            return train_dataloader, test_dataloader, valid_dataloader
        
        elif dataset_name == 'ptbxl-few-shot':
            print('Preparing dataloader for ptbxl-few-shot...')
            root_a = [config.root_ptbxl500 + config.TASKS_ptbxl[0] + config.folder_group[0]] * 6
            root_b = config.npy_files
            ptbxl500_path = [a + b for a, b in zip(root_a, root_b)]
            # train_data_path, train_labels_path, test_data_path, test_labels_path, val_data_path, val_labels_path = config.ptbxl500_path
            (train_data_path, train_labels_path,
            test_data_path, test_labels_path, 
            val_data_path, val_labels_path) = ptbxl500_path

            ptbdataset = dataset_map[dataset_name](train_data_path, train_labels_path)
            ptbdataset_val = dataset_map[dataset_name](val_data_path, val_labels_path)
            # ptbdataset = utils.clean_ecg_data(ptbdataset)
            # ptbdataset_val = utils.clean_ecg_data(ptbdataset_val)
            ptbdataset = ConcatDataset([ptbdataset, ptbdataset_val])
            few_shot_data = create_few_shot_unique_label_dataset(ptbdataset, num_samples_per_labelset=1, seed = 2025) # 1c1s
            # few_shot_data = create_few_shot_multilabel_dataset(ptbdataset, num_samples_per_class=1, seed = 2025) # 71 samples
            
            dataset_test = dataset_map[dataset_name](test_data_path, test_labels_path)
            # dataset_test = utils.clean_ecg_data(dataset_test)
            test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

            train_dataloader = DataLoader(few_shot_data, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
            valid_dataloader = None
            return train_dataloader, test_dataloader, valid_dataloader
        
        elif dataset_name == 'ptbxl-100':
            print('Preparing dataloader for ptbxl-100...')
            root_a = [config.root_ptbxl100 + config.TASKS_ptbxl[0] + config.folder_group[0]] * 6
            root_b = config.npy_files
            ptbxl500_path = [a + b for a, b in zip(root_a, root_b)]
            # train_data_path, train_labels_path, test_data_path, test_labels_path, val_data_path, val_labels_path = config.ptbxl500_path
            (train_data_path, train_labels_path,
            test_data_path, test_labels_path, 
            val_data_path, val_labels_path) = ptbxl500_path

            dataset_train = dataset_map[dataset_name](train_data_path, train_labels_path)
            dataset_test = dataset_map[dataset_name](test_data_path, test_labels_path)
            dataset_valid = dataset_map[dataset_name](val_data_path, val_labels_path)
            # dataset_train = utils.clean_ecg_data(dataset_train)
            # dataset_test = utils.clean_ecg_data(dataset_test)
            # dataset_valid = utils.clean_ecg_data(dataset_valid)
            dataset_train = ConcatDataset([dataset_train, dataset_valid])
            
            train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
            test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            return train_dataloader, test_dataloader, valid_dataloader
        
        elif dataset_name == 'g12ec':
            print('Preparing dataloader for g12ec...')
            # imputation很多实验在最后一个数据集分割上做的
            root_a = [config.root_g12ec + config.folder_group[0]] * 6
            root_b = config.npy_files
            g12ec_path = [a + b for a, b in zip(root_a, root_b)]
            (train_data_path, train_labels_path,
            test_data_path, test_labels_path, 
            val_data_path, val_labels_path) = g12ec_path

            dataset_train = dataset_map[dataset_name](train_data_path, train_labels_path)
            dataset_test = dataset_map[dataset_name](test_data_path, test_labels_path)
            dataset_valid = dataset_map[dataset_name](val_data_path, val_labels_path)
            # dataset_train = utils.clean_ecg_data(dataset_train)
            # dataset_test = utils.clean_ecg_data(dataset_test)
            # dataset_valid = utils.clean_ecg_data(dataset_valid)
            dataset_train = ConcatDataset([dataset_train, dataset_valid])
            
            train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
            test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            return train_dataloader, test_dataloader, valid_dataloader
            
        elif dataset_name == 'ludb':
            print('Preparing dataloader for ludb...')
            root_a = [config.root_ludb] * 6
            root_b = config.npy_files[:4]
            ludb_path = [a + b for a, b in zip(root_a, root_b)]
            (train_data_path, train_labels_path, test_data_path, test_labels_path) = ludb_path
            dataset_train = dataset_map[dataset_name](train_data_path, train_labels_path)
            dataset_test = dataset_map[dataset_name](test_data_path, test_labels_path)
            # dataset_train = ConcatDataset([dataset_train, dataset_test])
            train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
            test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            return train_dataloader, test_dataloader, 0

    elif mode == 'pretrain':
        if dataset_name is None:
            raise ValueError("Dataset name must be provided for pretrain mode.")
        
        dataset_map = {
            'mimic': mimic_ecgdataset,
        }
        
        if dataset_name == 'mimic':
            print('Preparing dataloader for mimic...Consdering the large size of the dataset, we divided the dataset into 6 pieces.')
            print('Preparing the 1-3 pieces.')
            # restricted to the cpu memory
            dataset1 = dataset_map[dataset_name](config.mimic_path[0], config.mimic_size[0])
            dataset2 = dataset_map[dataset_name](config.mimic_path[1], config.mimic_size[1])
            dataset3 = dataset_map[dataset_name](config.mimic_path[2], config.mimic_size[2])
            print('Preparing the 4-6 pieces.')
            dataset4 = dataset_map[dataset_name](config.mimic_path[3], config.mimic_size[3])
            dataset5 = dataset_map[dataset_name](config.mimic_path[4], config.mimic_size[4])
            dataset6 = dataset_map[dataset_name](config.mimic_path[5], config.mimic_size[5])
            
            dataset = ConcatDataset([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6])
            
            # total_size = len(dataset)
            # train_size = int(0.9 * total_size)
            # val_size = total_size - train_size
            # dataset, val_dataset = random_split(dataset, [train_size, val_size])

            # dataloader
            trian_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True)
            valid_dataloader = DataLoader(dataset6, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)
            return trian_dataloader, valid_dataloader
        
        elif dataset_name not in dataset_map:
            raise ValueError(f"Unsupported dataset name '{dataset_name}'. Available options are: {list(dataset_map.keys())}")
        
    # elif mode == 'classification':
    #     if dataset_name is None:
    #         raise ValueError("Dataset name must be provided for pretrain mode.")
    #     dataset_map = {
    #         'mimic': mimic_ecgdataset,
    #         'ptbxl-indices-500': ptbxldataset,
    #     }
        
    #     if dataset_name == 'ptbxl-indices-500':
    #         print('Preparing indices dataloader for ptbxl-500...')
    #         train_data_path, train_labels_path, test_data_path, test_labels_path = config.ptbxl500_indices_path

    #         dataset_train = dataset_map[dataset_name](train_data_path, train_labels_path)
    #         dataset_test = dataset_map[dataset_name](test_data_path, test_labels_path)

    #         train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    #         test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    #         return train_dataloader, test_dataloader
        
    #     elif dataset_name == 'ptbxl-feature-clf':
    #         print('Preparing dataloader for ptbxl-500...')
    #         train_data_path, train_labels_path, test_data_path, test_labels_path = config.ptbxl500_feature_path

    #         dataset_train = dataset_map[dataset_name](train_data_path, train_labels_path, 14520)
    #         dataset_test = dataset_map[dataset_name](test_data_path, test_labels_path, 1649)
            
    #         train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)
    #         test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    #         return train_dataloader, test_dataloader
        
    else:
        raise ValueError("Invalid mode. Please choose either 'pretrain' or 'finetune'.")