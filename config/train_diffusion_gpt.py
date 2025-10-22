# config for training diffusion-GPT (124M) down to very nice loss of ? on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train_diffusion_gpt.py config/train_diffusion_gpt.py

wandb_log = False
wandb_project = 'owt'
wandb_run_name='diffusion-gpt-124M'

tb_log = True

# these make the total batch size be ~0.5M
# 96 batch size * 1024 block size * 5 gradaccum * 1 GPUs = 491,520
batch_size = 96
block_size = 1024
gradient_accumulation_steps = 5

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 100
eval_iters = 10
log_interval = 10

# weight decay
weight_decay = 1e-1
