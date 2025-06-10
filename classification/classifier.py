'''
ref: 
[1] https://huggingface.co/docs/transformers/en/model_doc/resnet
[2] https://huggingface.co/docs/transformers/en/model_doc/swinv2
[3] https://github.com/seitalab/dnn_ecg_comparison
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinConfig, SwinModel
from transformers import ResNetConfig, ResNetModel

import utils.utils as utils
from utils.utils import Config
from dataloader import get_dataloader
from classification.baseline.resnet1d import resnet1d18
from classification.baseline.transformer import transformer_d2_h4_dim64l
from classification.functions.train_model import ModelTrainer as Trainer


# define head
class HeadModule(nn.Module):

    def __init__(self, in_dim: int, num_classes: int):
        super(HeadModule, self).__init__()

        self.fc1 = nn.Linear(in_dim, 512)
        # self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(0.2)
        # self.drop1 = nn.Dropout(0)

        self.fc3 = nn.Linear(512, num_classes)
        # self.fc3 = nn.Linear(512, 64)
        # self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size (num_batch, in_dim).
        Returns:
            feat (torch.Tensor): Tensor of size (num_batch, num_classes).
        """
        feat = F.relu(self.fc1(x))
        # feat = self.fc1(x) 
        feat = self.drop1(feat)
        feat = self.fc3(feat)
        # feat = self.bn1(feat)
        # feat = self.fc4(feat)
        return feat
    
class HeadModule_swin(nn.Module):

    def __init__(self, in_dim: int, num_classes: int):
        super(HeadModule_swin, self).__init__()

        self.fc1 = nn.Linear(in_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        # # self.drop1 = nn.Dropout(0.25)
        # self.drop1 = nn.Dropout(0)

        # self.fc3 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size (num_batch, in_dim).
        Returns:
            feat (torch.Tensor): Tensor of size (num_batch, num_classes).
        """
        # feat = F.relu(self.fc1(x))
        feat = self.fc1(x)
        # feat = self.drop1(feat)
        feat = self.bn1(feat)
        feat = self.fc3(feat)
        return feat

class Classifier(nn.Module):

    def __init__(self, backbone: nn.Module, prediction_heads: nn.Module):
        super(Classifier, self).__init__()

        self.backbone = backbone
        self.prediction_heads = prediction_heads

    def forward(self, x: torch.Tensor):

        feat = self.backbone(x)
        feat = (feat.pooler_output).view(x.shape[0], -1)
        predictions = self.prediction_heads(feat)

        return predictions


class resnet18_qpr(nn.Module):
    # qpr uses this for nonlinear classifications
    def __init__(self, num_classes: int):
        super(resnet18_qpr, self).__init__()
        configuration = ResNetConfig(num_channels = 12,
                                     depths = [2, 2, 2, 2],
                                     layer_type = 'basic')
        self.backbone = ResNetModel(configuration)

        # self.backbone = resnet1d18(num_lead=12)
        self.head = HeadModule(2048, num_classes)
        self.classifier = Classifier(self.backbone, self.head)

    def forward(self, x: torch.Tensor):
        return self.classifier(x)
    
    
class resnet18_baseline(nn.Module):
    def __init__(self, num_classes: int):
        super(resnet18_baseline, self).__init__()
        self.backbone = resnet1d18(num_lead=12)
        self.head = HeadModule(512, num_classes)
        self.classifier = Classifier(self.backbone, self.head)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        predictions = self.head(feat)
        return predictions


class transformerD2(nn.Module):
    def __init__(self, backbone_out_dim):
        super(transformerD2, self).__init__()
        self.backbone_out_dim = backbone_out_dim
        self.classifier = transformer_d2_h4_dim64l(chunk_len = 48, backbone_out_dim = self.backbone_out_dim)
        
    def forward(self, x: torch.Tensor):
        return self.classifier(x)
        
        
class swin_transformer(nn.Module):
    def __init__(self, num_classes):
        super(swin_transformer, self).__init__()
        swin_transformer_config = SwinConfig(image_size = 624,
                                            patch_size = 4,
                                            num_channels = 12,
                                            embed_dim = 64,
                                            depths=[2, 2, 6, 2],
                                            num_heads=[2, 4, 8, 16],
                                            output_hidden_states = True)

        self.num_classes = num_classes
        self.model = SwinModel(swin_transformer_config)
        self.head = HeadModule_swin(512, self.num_classes)
        
    def forward(self, x: torch.Tensor):
        output = self.model(x).pooler_output
        output = self.head(output)
        return output
        


class model_clf(nn.Module):
    # we embedding the indices of quantization and concat the quantization and indices
    def __init__(self, model_vqvae, classifier, embedding_dim=64):
        super(model_clf, self).__init__()
        self.model_vqvae = model_vqvae
        self.classifier = classifier
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Embedding(num_embeddings=1024, embedding_dim=self.embedding_dim)
        self.BN = nn.BatchNorm2d(12)

    def forward(self, data):
        z = self.model_vqvae._encoder(data) # z keys: 'last_hidden_state', 'loc', 'scale', 'patch_input'

        # z.last_hidden_state: (batch, 12, patches, patch_length)
        hidden_state = self.model_vqvae.BN(z.last_hidden_state)

        quantization, indices, loss = self.model_vqvae._vq(hidden_state) # (batch, 12, 625/500, 128)
        quantization = self.BN(quantization)

        # quantized_de = model_vqvae._map(quantization).reshape(data.shape[0], 12, -1) # batch, 12, 5000
        # if quantized_de.shape[-1] < 5000:
        #     quantized_de = torch.cat((quantized_de, quantized_de[:, :, -8:]), dim=-1)
        # quantized_de = model_vqvae.BN1d(quantized_de)
        # # quantized_de size: (batch, 12, 5000)

        # x_recon, _, _  = decoder_new(quantized_de.transpose(-1, -2))
        # # x_recon size: (batch, 5000, 12)
        # x_recon = x_recon * z.scale + z.loc # destandardization
        vq_loss = loss.sum() / self.model_vqvae.vq_heads
        perplexities = self.model_vqvae.perplexity_cal(indices, self.model_vqvae.num_embeddings, self.model_vqvae.vq_heads)
        
        index_embeddings_head1 = self.embedding_layer(indices[:, :, :, 0])
        index_embeddings_head2 = self.embedding_layer(indices[:, :, :, 1])
        index_embeddings_head3 = self.embedding_layer(indices[:, :, :, 2])
        index_embeddings_head4 = self.embedding_layer(indices[:, :, :, 3])
        combined_features = torch.cat((quantization, index_embeddings_head1, index_embeddings_head2, index_embeddings_head3, index_embeddings_head4), dim=-1)
        # combined_features: batch, 12, 624, 128+64*4

        y_pred = self.classifier(combined_features)
        return y_pred, vq_loss, perplexities



# class model_seg(nn.Module):
#     def __init__(self, model_vqvae):
#         super(model_seg, self).__init__()
#         self.model_vqvae = model_vqvae
#         self.segmentor = S4(d_input=12, d_model=3)

#     def forward(self, data):
#         z = self.model_vqvae._encoder(data) # z keys: 'last_hidden_state', 'loc', 'scale', 'patch_input'

#         # z.last_hidden_state: (batch, 12, patches, patch_length)
#         hidden_state = self.model_vqvae.BN(z.last_hidden_state)

#         quantization, indices, loss = self.model_vqvae._vq(hidden_state) # (batch, 12, 625/500, 128)

#         quantized_de = self.model_vqvae._map(quantization).reshape(data.shape[0], 12, -1) # batch, 12, 5000
#         if quantized_de.shape[-1] < 5000:
#             quantized_de = torch.cat((quantized_de, quantized_de[:, :, -8:]), dim=-1)
#         quantized_de = self.model_vqvae.BN1d(quantized_de) # 12, 5000
        
#         y_pred = self.segmentor(quantized_de.transpose(-1, -2)) # y_pred size: (batch, 5000, 4)
        
#         vq_loss = loss.sum() / self.model_vqvae.vq_heads
#         perplexities = self.model_vqvae.perplexity_cal(indices, self.model_vqvae.num_embeddings, self.model_vqvae.vq_heads)

#         return y_pred.permute(0, 2, 1), vq_loss, perplexities


def run(model, train_dataloader, test_dataloader, 
        epoch, lr, save_dir, log_dir, patience, eval_every, device):

    model = model.to(device)
    # train_loader = self._prepare_dataloader("train", is_train=True)
    # valid_loader = self._prepare_dataloader("val")
    trainer = Trainer(epochs=epoch, 
                      save_dir=save_dir, 
                      log_dir=log_dir, 
                      patience=patience, 
                      eval_every=eval_every, 
                      device=device)

    trainer.set_model(model)
    trainer.set_optimizer(lr)
    trainer.set_lossfunc()
    trainer.run(train_dataloader, test_dataloader)


if __name__ == "__main__":
    # if run this function, the baseline model will be trained.
    # and we train qpr on classification taks in main_qpr.py
    utils.set_seed(2025)

    config_path = 'config/vqvae.yaml'
    config = Config(config_path)

    train_dataloader, test_dataloader, _ = get_dataloader(mode='finetune', 
                                                          config=config, 
                                                          dataset_name='ptbxl-500', 
                                                          batch_size=64)
    
    print(len(train_dataloader), len(test_dataloader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifer = resnet18_baseline(71)
    # classifer = transformerD2(71)
    # classifer = swin_transformer(71)
    
    # resnet param：
    # transformer param：8, 5e-4
    run(classifer, 
        train_dataloader, 
        test_dataloader,
        epoch=100, 
        lr=1e-3, 
        save_dir='results/finetune_model_pth/clf-baseline/model', 
        log_dir='results/finetune_model_pth/clf-baseline/log/resnet', 
        patience=10, eval_every=2, 
        device=device)
