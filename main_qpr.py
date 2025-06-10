### main function for QPR training and finetuning
import sys
sys.path.append('/root/QPR4ECG/')

import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_dataloader, prepare_dataloader_multiclass
from QPR import qpr, LeadTransformer, train_vq, finetune_vq, prediction_vq, classification_ML, classification_MC, segmentation_vq
from classification.classifier import resnet18_qpr, model_clf, swin_transformer
from encoder.predictor import GPTIndexPredictor
from encoder.segmentor import model_seg
from utils.utils import Config
import utils.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### the cursor is used to optimize the project
def main(args):
    
    config_path = args.config
    config = Config(config_path)

    # pretrain stage
    if args.mode == 'pretrain':
        train_dataloader, valid_dataloader = get_dataloader(mode=args.mode, 
                                                            config=config, 
                                                            dataset_name=args.dataset_name, 
                                                            batch_size=args.batch_size)
        print(len(train_dataloader), len(valid_dataloader))

        model_vqvae = qpr(num_input_channeles=args.num_input_channeles,
                          context_length=args.context_length,
                          patch_length=args.patch_length,
                          patch_stride=args.patch_stride,
                          num_embeddings=args.num_embeddings, 
                          embedding_dim=args.embedding_dim, 
                          vq_form=args.vq_form, 
                          vq_heads=args.vq_heads, 
                          multi_proj=False).to(device)
        
        lead_transformer = LeadTransformer().to(device)
        
        train_vq(model_vqvae = model_vqvae, 
                 lead_transformer = lead_transformer,
                 train_dataloader = train_dataloader, 
                 lr = args.learning_rate, 
                 epoch = args.epoch, 
                 device = device, 
                 aug_gaussian = True, 
                 Einthoven_loss = True, 
                 log_path = args.log_path_pretrain, 
                 save_model_dir = args.pretrain_model_save_path,
                 checkpoint_path = args.checkpoint_path)
        
        # model_vqvae = torch.load('/root/VQ4ECG/results/model_pth/vqvae_iter_6.pth', map_location=device)
        # lead_transformer = torch.load('/root/VQ4ECG/results/model_pth/lead_transformer_iter_6.pth', map_location=device)
        
        # eval_usage_fitting(model_vqvae=model_vqvae, 
        #                    valid_dataloader=valid_dataloader, 
        #                    mode=args.mode, 
        #                    num_embeddings=args.num_embeddings, 
        #                    device=device, 
        #                    rd_index=0)
        
        print(f'Pretrained ok, model is saved: {args.pretrain_model_save_path}')

    elif args.mode == 'finetune':
        # finetune on ptbxl
        if args.dataset_name in ['ptbxl-500', 'ptbxl-few-shot', 'g12ec', 'ludb']:
            utils.set_seed(2025)
            if args.downstream_task != 'classification-mc':
                train_dataloader, test_dataloader, valid_dataloader = get_dataloader(mode=args.mode, 
                                                                                    config=config, 
                                                                                    dataset_name=args.dataset_name, 
                                                                                    batch_size=args.batch_size)
                print(len(train_dataloader), len(test_dataloader))
            
            if args.downstream_task == 'classification': # 默认multi-label
                model_vqvae = torch.load(config.classification_qpr, map_location = device)
                decoder_new = resnet18_qpr(71)
                # decoder_new = swin_transformer(71) # 71,44,23,5,19,12
                model_clf_concat = model_clf(model_vqvae, decoder_new, embedding_dim=64).to(device)
                
                assert model_vqvae._encoder.masking.random_mask_ratio == 0
                classification_ML(model_clf = model_clf_concat,
                                  train_dataloader = train_dataloader, 
                                  test_dataloader = test_dataloader,
                                  lr = args.learning_rate, 
                                  epoch = args.epoch, 
                                  device = device, 
                                  Einthoven_loss = False, 
                                  log_path = config.classification_log_path, 
                                  save_model_dir = config.classification_model_save_path,
                                  checkpoint_path = config.checkpoint_path)
                
                print(f'Task ok! model saved in: {config.finetune_model_save_path}')

            elif args.downstream_task == 'classification-mc':
                train_dataloader, test_dataloader, weight = prepare_dataloader_multiclass(config=config,
                                                                                          task_name=args.MC_task_name,
                                                                                          dataset_name=args.dataset_name,
                                                                                          batch_size=args.batch_size,
                                                                                          frequency=500, 
                                                                                          length=10)
                print(len(train_dataloader), len(test_dataloader))
                weight = torch.Tensor(weight).to(device)
                model_vqvae = torch.load(config.classification_qpr, map_location = device)
                # decoder_new = resnet18_qpr(30)
                decoder_new = swin_transformer(3) # 71,44,23,5,19,12
                model_clf_concat = model_clf(model_vqvae, decoder_new, embedding_dim=64).to(device)

                assert model_vqvae._encoder.masking.random_mask_ratio == 0
                classification_MC(model_clf = model_clf_concat,
                                  train_dataloader = train_dataloader, 
                                  test_dataloader = test_dataloader,
                                  weight = weight,
                                  lr = args.learning_rate, 
                                  epoch = args.epoch, 
                                  device = device, 
                                  log_path = config.classification_log_path, 
                                  save_model_dir = config.classification_model_save_path,
                                  checkpoint_path = config.checkpoint_path)
                
                print(f'Multi-classification task ok! model saved in: {config.finetune_model_save_path}')
                
            elif args.downstream_task == 'finetune':
                model_vqvae = torch.load(config.finetune_pretrained_qpr, map_location = device)
                predictionhead = nn.Identity()
                lead_transformer = torch.load(config.finetune_pretrained_lead_transformer, map_location = device)
                model_vqvae._encoder.masking.random_mask_ratio = 0
                
                finetune_vq(signal_length=5000,
                            model_vqvae = model_vqvae, 
                            lead_transformer = lead_transformer,
                            predictionHead = predictionhead,
                            train_dataloader = train_dataloader, 
                            lr = args.learning_rate, 
                            epoch = args.epoch, 
                            device = device, 
                            Einthoven_loss = True, 
                            log_path = config.finetune_log_path, 
                            save_model_dir = config.finetune_model_save_path,
                            checkpoint_path = config.checkpoint_path)
                
                print(f'Task ok! model saved in: {config.finetune_model_save_path}')
                
            elif args.downstream_task == 'imputation':
                model_vqvae = torch.load(config.imputation_pretrained_qpr, map_location = device)
                # model_vqvae._encoder.masking.channel_consistent_masking = True
                model_vqvae._encoder.masking.random_mask_ratio = 0.4
                predictionhead = nn.Identity()
                lead_transformer = torch.load(config.imputation_pretrained_lead_transformer, map_location = device)
                
                finetune_vq(signal_length=5000,
                            model_vqvae = model_vqvae, 
                            lead_transformer = lead_transformer,
                            predictionHead = predictionhead,
                            train_dataloader = train_dataloader, 
                            lr = args.learning_rate, 
                            epoch = args.epoch, 
                            device = device, 
                            Einthoven_loss = True, 
                            log_path = config.imputation_log_path, 
                            save_model_dir = config.imputation_model_save_path,
                            checkpoint_path = config.checkpoint_path)
                
                print(f'Task ok! model saved in: {config.finetune_model_save_path}')
            
            elif args.downstream_task == 'prediction' or 'forecast':
                model_vqvae = torch.load(config.prediction_finetuned_qpr, map_location = device)
                # model_prediction = torch.load(config.prediction_finetuned_lead_transformer, map_location = device)
                model_prediction = GPTIndexPredictor().to(device)
                # predictionhead = prediction_qpr()
                # model_prediction = model_pred(model_vqvae, predictionhead).to(device)
                
                prediction_vq(model_vqvae = model_vqvae, 
                              model_pred = model_prediction,
                              train_dataloader = train_dataloader, 
                              test_dataloader = test_dataloader,
                              lr = args.learning_rate, 
                              epochs = args.epoch, 
                              device = device, 
                              log_path=config.prediction_log_path, 
                              save_model_dir = config.prediction_model_save_path)
                print(f'Task ok! model saved in: {config.finetune_model_save_path}')
                
            elif args.downstream_task == 'segmentation':
                model_vqvae = torch.load(config.segmentation_qpr, map_location = device)
                model_segentation = model_seg(model_vqvae).to(device)
                segmentation_vq(model_seg = model_segentation,
                                train_dataloader = train_dataloader, 
                                test_dataloader = test_dataloader,
                                lr = args.learning_rate, 
                                epoch = args.epoch, 
                                device = device, 
                                log_path = config.segmentation_log_path, 
                                save_model_dir = config.segmentation_model_save_path,
                                checkpoint_path = config.checkpoint_path)
                
            print(f'Task ok! model saved in: {config.finetune_model_save_path}')

    else:
        raise ValueError(f"No supported mode: {args.mode}")


if __name__ == "__main__":
    
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description="VQ-VAE Training and Finetuning")

    # config and dataset description
    parser.add_argument('--mode', type=str, default='pretrain', choices=['pretrain', 'finetune'], help='training stage')
    parser.add_argument('--downstream_task', type=str, default='classification', choices=['classification', 'classification-mc', 'imputation', 'forecast', 'segmentation'], help='downstream task')
    parser.add_argument('--config', type=str, default='config/vqvae.yaml', help='vqvae-config-path')
    parser.add_argument('--dataset_name', type=str, default='mimic', choices=['mimic', 'ptbxl-100', 'ptbxl-500', 'ptbxl-full', 'ptbxl-few-shot', 'g12ec', 'ludb'], 
                        help='dataset_name')
    parser.add_argument('--MC_task_name', type=str, default='AF', choices=['AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC'])


    # train paramaters
    parser.add_argument('--batch_size', type=int, default=64, help="batch size") # classification-8
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--epoch', type=int, default=100, help="number of training updates")
    
    # qpr parameters
    parser.add_argument('--num_input_channeles', type=int, default=12, help="number of input channels")
    parser.add_argument('--context_length', type=int, default=5000, help="context length")
    parser.add_argument('--patch_length', type=int, default=16, help="patch length")
    parser.add_argument('--patch_stride', type=int, default=8, help="patch stride")
    
    # vq parameters
    parser.add_argument('--num_embeddings', type=int, default = 1024, help="codebook size")
    parser.add_argument('--embedding_dim', type=int, default = 128, help="vector quantization shape")
    parser.add_argument('--vq_heads', type=int, default = 4, help="number of vector quantizers")
    parser.add_argument('--vq_form', type=str, default='residualsim', choices=['normal', 'residual', 'residualsim', 'simvq'], help="different quantization method")
    
    # # tesnorboard/checkpoint/model save
    # parser.add_argument('--log_path_pretrain', type=str, default='results/log/mimic/experiment-6',
    #                     help="pretrain-log")
    # parser.add_argument('--log_path_finetune', type=str, default='results/log/ptbxl-100/finetune-forecast-1',
    #                     help="finetune-log")
    # parser.add_argument('--log_path_classification', type=str, default='results/log/ptbxl-100/finetune-forecast-1',
    #                     help="classification-log")
    
    # parser.add_argument('--checkpoint_path', type=str, default='results/checkpoint/all_data_test.pth',
    #                     help="checkpoint-path")
    
    # parser.add_argument('--pretrain_model_save_path', type=str, default='results',
    #                     help="pretrain model save")
    # parser.add_argument('--pretrained_vqvae', type=str, default='results/pretrain_model_pth/experiment-15_vqvae_final.pth',
    #                     help="pretrained vqvae")
    # parser.add_argument('--pretrained_lead_transformer', type=str, default='results/pretrain_model_pth/experiment-15_lead_transformer_final.pth',
    #                     help="pretrained lead transformer")
    # parser.add_argument('--finetune_model_save_path', type=str, default='results/finetune_model_pth/ptbxl-100/finetune-forecast-1',  
    #                     help="finetune model save")
    

    args = parser.parse_args()
    main(args)