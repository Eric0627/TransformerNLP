# add all your Encoder and Decoder code here
import torch
import torch.nn as nn
from torch.nn import functional as F

class EncoderWithClassifier(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, device, n_hidden, n_output, dropout=0.0):
        super().__init__()

        self.encoder = Encoder(vocab_size, n_embd, block_size, n_head, n_layer, device, dropout=dropout)
        self.classifier = FeedForwardClassifier(n_embd, n_hidden, n_output)
        
    def forward(self, idx, targets = None):
        embedding = self.encoder(idx)
        embedding = embedding.mean(dim=1)
        logits = self.classifier(embedding)
        if targets is None:
                loss = None
        else:
            loss = F.cross_entropy(logits, targets) # cross_entropy() applies the softmax function internally
        logits = F.softmax(logits, dim=1)
        return logits, loss

    def num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

class Encoder(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, device, dropout=0.0):
        super().__init__()
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # embedding table for the tokens
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # embedding table for the positions
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, n_hidden=256, dropout=dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd) # the last layer norm

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln(x)
        return x
    
    def get_attention_maps(self, idx):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        attention_maps = []
        for l in self.blocks:
            _, attn_map = l.sa(x, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

class FeedForwardClassifier(nn.Module):
    def __init__(self, n_embd, hidden_size, output_size):
        super(FeedForwardClassifier, self).__init__()
        self.fc1 = nn.Linear(n_embd, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.softmax(x, dim=1) # Apply Softmax to obtain output probabilities.
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, device):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, n_hidden=100, mask=True) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd) # # the last layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        # logits = F.softmax(logits, dim=-1)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # cross_entropy() applies the softmax function internally
        return logits, loss

    def num_params(self):
       n_params = sum(p.numel() for p in self.parameters())
       return n_params
    
    def get_attention_maps(self, idx):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        attention_maps = []
        for l in self.blocks:
            _, attn_map = l.sa(x, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, block_size, mask = False, dropout = 0.0):
        super().__init__()
        self.mask = mask
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # lower triangluar matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape # batch size, block size, channels
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities") scaled by d_k
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        if self.mask == True: # mask the future tokens
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out, wei
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, n_head, block_size, mask = False, dropout = 0.0):
        super().__init__()
        head_size = n_embd // n_head # n_embd % n_head == 0
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, mask=mask, dropout=dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention = False):
        # Concatenate outputs from all heads and project the output with a linear layer
        out = torch.cat([h(x)[0] for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        attention_weight = torch.cat([h(x)[1] for h in self.heads], dim=0)
        if return_attention:
            return out, attention_weight
        else:
            return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, n_hidden, dropout = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size, n_hidden, mask = False, dropout = 0.0):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, mask=mask, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, n_hidden, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd) # layer normalization
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual connection + layer normalization
        x = x + self.ffwd(self.ln2(x))
        return x


class MultiQueryAttention(nn.Module):
    # reference: https://arxiv.org/pdf/1911.02150
    """ Multi-query attention """

    def __init__(self, n_embd, block_size, n_query = 2, mask = False, dropout = 0.0):
        super().__init__()
        self.n_query = n_query
        self.mask = mask
        n_out = n_embd // 8
        self.key = nn.Linear(n_embd, n_out, bias=False)
        self.value = nn.Linear(n_embd, n_out, bias=False)
        self.query = nn.ModuleList([nn.Linear(n_embd, n_out, bias=False) for _ in range(n_query)])
        # lower triangluar matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout1 = nn.Dropout(dropout)
        self.proj = nn.Linear(n_query * n_out, n_embd)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape # batch size, block size, channels
        k = self.key(x) # (B,T,C)
        v = self.value(x) # (B,T,C)
        outputs = []
        for i in range(self.n_query):
            q = self.query[i](x) # (B,T,C)
            # compute attention scores ("affinities") scaled by d_k
            wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
            if self.mask == True: # mask the future tokens
                wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
            wei = F.softmax(wei, dim=-1) # (B, T, T)
            wei = self.dropout1(wei)
            # perform the weighted aggregation of the values
            out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
            outputs.append(out)
        out = torch.cat(outputs, dim=2)
        out = self.dropout2(self.proj(out))
        return out

class Block_MQA(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, block_size, n_hidden, mask = False):
        super().__init__()
        self.mqa = MultiQueryAttention(n_embd, block_size, mask=mask)
        self.ffwd = FeedFoward(n_embd, n_hidden)
        self.ln1 = nn.LayerNorm(n_embd) # layer normalization
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.mqa(self.ln1(x)) # residual connection + layer normalization
        x = x + self.ffwd(self.ln2(x))
        return x

class Decoder_MQA(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer, device):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block_MQA(n_embd, block_size, n_hidden=100, mask=True) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd) # # the last layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        # logits = F.softmax(logits, dim=-1)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # cross_entropy() applies the softmax function internally
        return logits, loss

    def num_params(self):
       n_params = sum(p.numel() for p in self.parameters())
       return n_params
