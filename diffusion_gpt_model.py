import math
import inspect
from dataclasses import dataclass
import warnings

import torch
import torch.nn as nn
from torch.nn import functional as F

from gpt2_model import LayerNorm, MLP

@dataclass
class DiffusionGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    n_lm_heads: int = 1 # p(x | x^t) = \sum_k \pi_k \prod_i p^k(x_i | x^t)
    cce_impl: str = 'none'

# copy-paste CausalSelfAttention from model.py, but remove causal masking
class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# copy-paste Block from model.py, but replace CausalSelfAttention with SelfAttention
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class DiffusionGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size + 1, config.n_embd), # add CLS token
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        if config.n_lm_heads == 1:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            # with weight tying when using torch.compile() some warnings get generated:
            # "UserWarning: functional_call was passed multiple values for tied weights.
            # This behavior is deprecated and will be an error in future versions"
            # not 100% sure what this is, so far seems to be harmless. TODO investigate
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        else:
            self.lm_heads = nn.ModuleList([
                nn.Linear(config.n_embd, config.vocab_size, bias=False)
                for _ in range(config.n_lm_heads)
            ])
            self.pi_head = nn.Linear(config.n_embd, config.n_lm_heads)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    @property
    def mask_token_id(self):
        return self.config.vocab_size - 2

    @property
    def cls_token_id(self):
        return self.config.vocab_size - 1

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, target_ids=None, mask=None, p_mask=None):
        """
        ``input_ids`` - masked input ids
        ``target_ids`` - unmasked input ids (not shifted)
        ``mask`` - True if masked, else False
        ``p_mask`` - tensor of shape (b,) with masking probabilities for each text in the batch, needed for loss computation
        """
        device = input_ids.device
        b, l = input_ids.size()
        assert l <= self.config.block_size, f"Cannot forward sequence of length {l}, block size is only {self.config.block_size}"

        # append CLS token
        cls_token_ids = torch.full(size=(b, 1), fill_value=self.cls_token_id, dtype=input_ids.dtype, device=device)
        input_ids = torch.cat((cls_token_ids, input_ids), dim=1)
        pos = torch.arange(0, l + 1, dtype=torch.long, device=device) # shape (l + 1)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, l + 1, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (l + 1, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if target_ids is not None:
            assert mask is not None
            assert p_mask is not None

            if self.config.n_lm_heads == 1:
                x = x[:, 1:].contiguous()  # drop CLS token, we do not need it in the basic case
                if self.config.cce_impl == 'none':
                    logits = self.lm_head(x) # (b, l, vocab_size)
                    logits[:, :, self.mask_token_id] = float('-inf') # zero prob of MASK token
                    logits[:, :, self.cls_token_id] = float('-inf') # zero prob of CLS token
                    nll = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction='none').view(b, l)
                else:
                    # significantly reduces memory footprint, see https://arxiv.org/abs/2411.09009v2
                    from cut_cross_entropy import linear_cross_entropy
                    w = self.lm_head.weight[:-2] # remove logits corresponding to MASK and CLS tokens
                    nll = linear_cross_entropy(x.view(-1, x.size(-1)), w, target_ids.view(-1),
                                               reduction='none', impl=self.config.cce_impl).view(b, l)
                nll = (nll * mask).sum(1) # carry-over masking
            else:
                with warnings.catch_warnings():
                    # using log_softmax with torch.compile causes an annoying warning for some reason
                    warnings.filterwarnings("ignore", message=".*Online softmax is disabled.*")
                    log_pi = torch.log_softmax(self.pi_head(x[:, 0]), dim=-1) # (b, n_lm_heads)
                x = x[:, 1:].contiguous()
                nll_per_head = []
                for lm_head in self.lm_heads:
                    if self.config.cce_impl == 'none':
                        logits = lm_head(x) # (b, l, vocab_size)
                        logits[:, :, self.mask_token_id] = float('-inf') # zero prob of MASK token
                        logits[:, :, self.cls_token_id] = float('-inf') # zero prob of CLS token
                        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction='none').view(b, l)
                    else:
                        from cut_cross_entropy import linear_cross_entropy
                        w = lm_head.weight[:-2] # remove logits corresponding to MASK and CLS tokens
                        nll = linear_cross_entropy(x.view(-1, x.size(-1)), w, target_ids.view(-1),
                                                   reduction='none', impl=self.config.cce_impl).view(b, l)
                    nll = (nll * mask).sum(1) # carry-over masking
                    nll_per_head.append(nll)
                nll_per_head = torch.stack(nll_per_head, dim=1) # (b, n_lm_heads)
                nll = -torch.logsumexp(log_pi - nll_per_head, dim=1) # (b,)

            # division by p_mask below follows from math (see ELBO in https://arxiv.org/pdf/2406.07524),
            # but it also makes perfect sense for the following general reason:
            # nll loss is sum over masked tokens (for unmasked tokens we have zero loss due to carry-over masking)
            # for large p_mask, many tokens are masked and nll loss is larger
            # for small p_mask, a few tokens are masked and nll loss is smaller
            # division by p_mask rescales nll loss, such that the model is equally penalized for all p_mask from [0, 1] range
            loss = torch.mean(nll / p_mask) / l
        else:
            loss = None

        return x, loss

    @torch.no_grad
    def compute_ppl(self, idx, max_steps=1024):
        raise NotImplementedError

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, n_steps):
        raise NotImplementedError
