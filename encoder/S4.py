from encoder.s4.S4D import S4D
import torch.nn as nn

# class Encoder(nn.Module):
#     def __init__(
#         self,
#         d_input = 12,
#         d_model = 64,
#         n_layers = 6,
#         dropout = 0.2,
#         d_state = 64,
#         prenorm = False,
#     ):
#         """
#         Args:
#         d_input:输入维度
#         d_state:SSM state expansion factor 隐藏状态
#         n_layers:S4D层数
#         prenorm：在S4D之前做LN OR 在S4D之后做LN
#         """
#         super().__init__()

#         self.prenorm = prenorm

#         self.encoder = nn.Linear(d_input, d_model)

#         self.s4_layers = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         self.dropouts = nn.ModuleList()

#         for _ in range(n_layers):
#             self.s4_layers.append(
#                 S4D(d_model, dropout=dropout, transposed=True,d_state = d_state,lr=min(0.001, 0.01))
#             )
#             self.norms.append(nn.LayerNorm(d_model))
#             self.dropouts.append(nn.Dropout1d(dropout))


#     def forward(self, x):
#         """
#         Input x is shape (B, L, C) -- (b , 1000, 12)
#         outputs shape (B, L, d_model)
#         """

#         x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

#         x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        
#         # for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
#         for layer, norm in zip(self.s4_layers, self.norms):

#             z = x
#             if self.prenorm:
#                 # Prenorm
#                 z = norm(z.transpose(-1, -2)).transpose(-1, -2)

#             # S4
#             z, _ = layer(z)

#             # Dropout
#             # z = dropout(z)

#             # Residual
#             x = z + x

#             if not self.prenorm:

#                 x = norm(x.transpose(-1, -2))

#         # x = x.transpose(-1, -2) # (B, d_model, L)->(B, L, d_model)

#         return x
    
# SSM for predictionHead and Decoder(for pretraining)
class S4(nn.Module):
    def __init__(
        self,
        d_input,
        d_model,
        n_layers = 6,
        dropout = 0.1,
        d_state = 64,
        prenorm = False,
    ):
        """
        Args:
        input size: (batch, 1000/5000, 12)
        d_input: input channel size
        d_state: SSM state expansion factor (hidden state)
        n_layers: number of S4D layers
        """
        super(S4, self).__init__()
        # pdb.set_trace()

        self.d_input = d_input
        self.d_model = d_model
        self.pre_proj = nn.Linear(self.d_input, self.d_model)
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(n_layers):
            self.s4_layers.append(
                S4D(self.d_model, dropout=dropout, transposed=True, d_state=d_state, lr=min(0.001, 0.01))
            )
            if i % 2 == 0:
                self.norms.append(nn.BatchNorm1d(self.d_model))
            else:
                self.norms.append(None)
            self.dropouts.append(nn.Dropout1d(dropout))

    def forward(self, x):
        # x size: (batch, 1000/5000, channel)
        x = self.pre_proj(x) 
        x = x.transpose(-1, -2)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
        # for layer, norm in zip(self.s4_layers, self.norms):
            z, _ = layer(x)
            # Dropout
            z = dropout(z)
            x = z + x
            if norm is not None:
                # x = norm(x.transpose(-1, -2)).transpose(-1, -2)
                x = norm(x)
            
        return x.transpose(-1, -2) # batch, 1000/5000, channel