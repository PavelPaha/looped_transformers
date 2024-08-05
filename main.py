import torch
from pipelines import start_training

from config import Args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device = {device}')

args = [
    Args(name='gpt2', family="gpt2", n_layer=24, epochs=50),
    Args(name='gpt2_loop_24_50', family="gpt2_loop", n_loop_window=24, epochs=50),
    Args(name='gpt2_loop_12_100', family="gpt2_loop", n_loop_window=12, epochs=100),
    Args(name='gpt2_loop_48_25', family="gpt2_loop", n_loop_window=48, epochs=25),
    Args(name='gpt2_cutted_24_50', family="gpt2_cutted", n_loop_window=24, epochs=50),
    Args(name='gpt2_cutted_12_100', family="gpt2_cutted", n_loop_window=12, epochs=100),
    Args(name='gpt2_cutted_48_25', family="gpt2_cutted", n_loop_window=48, epochs=25),
    Args(name='gpt2_cyclic_5_2_50', family="gpt2_cyclic", n_loop_window=5, block_size=2, epochs=50),
    Args(name='gpt2_cyclic_2_5_50', family="gpt2_cyclic", n_loop_window=2, block_size=5, epochs=50)
]

for arg in args:
    print(f'Start {arg.name}')
    start_training(arg, device)
