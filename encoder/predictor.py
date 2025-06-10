import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GPTIndexPredictor(nn.Module):
    def __init__(self, 
                 codebook_size, 
                 d_model, 
                 nhead, 
                 num_layers, 
                 num_channels, 
                 num_codebooks):
        super().__init__()
        self.d_model = d_model
        self.num_codebooks = num_codebooks
        self.num_channels = num_channels

        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, d_model) for _ in range(num_codebooks)
        ])

        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        self.heads = nn.ModuleList([
            nn.Linear(d_model, codebook_size) for _ in range(num_codebooks)
        ])
        
        # 缓存用于提高自回归生成效率
        self.cached_attention = None

    def forward(self, x, mask=None, use_cache=False):
        """
        前向传播方法
        
        Args:
            x: 输入张量 [batch_size, num_channels, seq_len, num_codebooks]
            mask: 注意力掩码
            use_cache: 是否使用缓存提高自回归生成效率
            
        Returns:
            各个位置的logits [batch_size, num_channels, seq_len, num_codebooks, codebook_size]
        """
        batch_size, _, seq_len, _ = x.shape
        
        # 重塑输入以便于处理
        x = x.view(-1, seq_len, self.num_codebooks)

        # 合并各个码本的嵌入
        embeddings = []
        for i in range(self.num_codebooks):
            emb = self.embeddings[i](x[..., i])
            embeddings.append(emb)
        embedded = sum(embeddings)  # 各码本嵌入求和

        # 应用位置编码
        embedded = self.pos_encoder(embedded) * math.sqrt(self.d_model)
        embedded = embedded.permute(1, 0, 2)  # [seq_len, batch*channels, d_model]
        
        # 创建自回归掩码
        if mask is None:
            mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        # 使用Transformer处理序列
        output = self.transformer(
            embedded, embedded,
            tgt_mask=mask,
            memory_mask=mask
        )
        
        # 重新排列输出维度
        output = output.permute(1, 0, 2)  # [batch*channels, seq_len, d_model]

        # 计算每个码本的logits
        logits = [head(output) for head in self.heads]
        logits = torch.stack(logits, dim=-2)  # [batch*channels, seq_len, num_codebooks, codebook_size]
        
        # 重塑回原始维度顺序
        return logits.view(batch_size, self.num_channels, seq_len, self.num_codebooks, -1)

    def generate_square_subsequent_mask(self, sz):
        # 创建自回归掩码矩阵
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def predict(self, input_indices, predict_steps=63, temperature=1.0, top_k=0, top_p=0.0):
        """
        使用自回归方式生成新的索引序列
        
        Args:
            input_indices: 输入序列索引 [batch_size, num_channels, seq_len, num_codebooks]
            predict_steps: 要生成的步数
            temperature: 控制采样随机性，较低值产生更确定的输出
            top_k: 如果>0，只保留概率最高的top-k个token
            top_p: 如果>0，使用nucleus sampling (按累积概率筛选token)
            
        Returns:
            生成的索引序列
        """
        self.eval()
        with torch.no_grad():
            current_seq = input_indices
            
            for _ in range(predict_steps):
                # 计算当前序列的logits
                logits = self(current_seq)
                next_logits = logits[:, :, -1]  # [batch, channels, num_codebooks, vocab_size]
                
                # 应用温度缩放
                if temperature != 1.0:
                    next_logits = next_logits / temperature
                
                next_indices_list = []
                for i in range(self.num_codebooks):
                    codebook_logits = next_logits[..., i, :]
                    
                    # 应用top-k过滤
                    if top_k > 0:
                        values, _ = torch.topk(codebook_logits, top_k)
                        min_values = values[..., -1].unsqueeze(-1).expand_as(codebook_logits)
                        codebook_logits = torch.where(codebook_logits < min_values, 
                                                    torch.full_like(codebook_logits, float('-inf')), 
                                                    codebook_logits)
                    
                    # 应用top-p (nucleus) 采样
                    if top_p > 0.0:
                        sorted_logits, sorted_indices = torch.sort(codebook_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # 移除累积概率超过阈值的token
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # 保证至少有一个token可用
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # 将排序的索引映射回原始索引
                        indices_to_remove = torch.zeros_like(codebook_logits, dtype=torch.bool)
                        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
                        codebook_logits[indices_to_remove] = float('-inf')
                    
                    # 根据概率分布进行采样或使用argmax
                    if temperature > 0:
                        # 采样模式
                        probs = F.softmax(codebook_logits, dim=-1)
                        next_token = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1)
                        next_token = next_token.reshape(probs.size(0), probs.size(1))
                    else:
                        # 贪婪模式
                        next_token = codebook_logits.argmax(-1)
                    
                    next_indices_list.append(next_token.unsqueeze(-1))
                
                # 将所有码本的预测结果合并
                next_indices = torch.cat(next_indices_list, dim=-1)
                
                # 将新预测的索引添加到序列中
                current_seq = torch.cat([
                    current_seq, 
                    next_indices.unsqueeze(2)
                ], dim=2)

            return current_seq[:, :, -predict_steps:]