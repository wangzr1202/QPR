# Paper under review



## Datasets

| Dataset      | Task                                   | Download link                                                |
| ------------ | -------------------------------------- | ------------------------------------------------------------ |
| MIMIC-IV-ECG | pre-train                              | https://physionet.org/content/mimic-iv-ecg/1.0/              |
| PTB-XL       | imputation, prediction, classification | https://physionet.org/content/ptb-xl/1.0.3/                  |
| G12EC        | imputation, classification             | https://www.kaggle.com/datasets/bjoernjostein/georgia-12lead-ecg-challenge-database |
| LUDB         | segmentation                           | https://www.physionet.org/content/ludb/1.0.1/                |

Total storage space requirement: >100GB

## Environment

```
pip install -r requirements.txt
```

## Usage

We use `ArgumentParser` to record training parameters, and store model paths in qpr.yaml.

### pre-train

```
cd QPR4ECG
```



```
python main_qpr.py \
    --mode pretrain \
    --config config/qpr.yaml \
    --dataset_name mimic \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --epoch 100 \
    --num_input_channeles 12 \
    --context_length 5000 \
    --patch_length 16 \
    --patch_stride 8 \
    --num_embeddings 1024 \
    --embedding_dim 128 \
    --vq_heads 4 \
    --vq_form residualsim
```

 ### fine-tune

```
cd QPR4ECG
```



```
python main_qpr.py \
    --mode finetune \
    --downstream_task classification \
    --config config/vqvae.yaml \
    --dataset_name ptbxl-500 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --epoch 50 \
    --num_input_channeles 12 \
    --context_length 5000 \
    --patch_length 16 \
    --patch_stride 8 \
    --num_embeddings 1024 \
    --embedding_dim 128 \
    --vq_heads 4 \
    --vq_form residualsim
```

