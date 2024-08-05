import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from nano_gpt import GPT2Config, GPT2Model


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, n_last_tokens=10, pred_type='regression'):
        super(TransformerModel, self).__init__()
        self.freq = 2
        self.ind = 0
        configuration = GPT2Config()
        configuration.block_size = self.freq * (n_positions + 1)  # +1 for xtest
        configuration.n_layer = n_layer
        configuration.n_head = n_head
        configuration.n_embd = n_embd
        configuration.dropout = 0.0
        configuration.bias = True
        configuration.n_last_tokens = n_last_tokens
        self.configuration = configuration

        self.n_last_tokens = n_last_tokens
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.n_layer = n_layer
        self._pred_type = pred_type

        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(self.configuration)
        self._read_out = nn.Linear(n_embd, 1)

        self.print_flag = False

    def _combine(self, xs_b, ys_b):
        B, n, d = xs_b.shape
        device = xs_b.device
        ys_b_wide = torch.cat((ys_b.view(B, n, 1), torch.zeros(B, n, d - 1, device=device)), axis=2)
        zs = torch.stack((xs_b, ys_b_wide), dim=2).view(B, self.freq * n, d)
        return zs

    def forward(self, xs, ys):
        B, n, d_in = xs.shape
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        f_output = self._backbone(inputs_embeds=embeds)
        prediction = self._read_out(f_output)
        y = prediction[:, self.ind::self.freq, 0]
        return y


class TransformerModelTying(TransformerModel):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModelTying, self).__init__(n_dims, n_positions, n_embd, n_layer, n_head)
        self.configuration.n_layer = 1
        self._backbone = GPT2Model(self.configuration)

    def f(self, output):
        return self._backbone(inputs_embeds=output)

    def forward(self, xs, ys):
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = embeds
        for idx in range(self.n_layer):
            output = self.f(output)
        prediction = self._read_out(output)
        y = prediction[:, self.ind::self.freq, 0]
        return y


class CyclicBlockTransformer(TransformerModel):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, block_size=3, n_loops=3,
                 pred_type='regression'):
        super(CyclicBlockTransformer, self).__init__(n_dims, n_positions, n_embd, n_layer, n_head,
                                                     pred_type)
        self.block_size = block_size
        self.n_loops = n_loops
        self.blocks = nn.ModuleList([GPT2Model(self.configuration) for _ in range(block_size)])

    def forward(self, xs, ys):
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)

        def custom_forward(x):
            return self.blocks[i % self.block_size](inputs_embeds=x)

        output = embeds
        for i in range(self.n_loops):
            output = checkpoint(custom_forward, output)

        prediction = self._read_out(output)
        y = prediction[:, self.ind::self.freq, 0]
        return y

def zero_out_positions(tensor, num_positions_to_zero):
    assert num_positions_to_zero <= tensor.size(1), "Number of positions to zero out cannot be more than tensor size."
    size = tensor.size(1)
    mask = torch.ones(size, dtype=torch.bool)
    zero_indices = torch.randperm(size)[:num_positions_to_zero]
    mask[zero_indices] = False

    masked_tensor = tensor.clone()
    masked_tensor[:, mask] = 0
    return masked_tensor


class TransformerModelLooped(TransformerModel):
    def __init__(
            self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, n_last_tokens=10, loop_func='z=f(x+z)',
            pred_type='regression', n_loops=10):
        super(TransformerModelLooped, self).__init__(
            n_dims, n_positions, n_embd, n_layer, n_head, n_last_tokens, pred_type)
        self.loop_func = loop_func
        self.n_loops = n_loops

    def f(self, output, embeds):
        if self.loop_func == 'z=f(x+z)':
            f_output = self._backbone(inputs_embeds=output + embeds)  # [B, 2n + 1, d]
        elif self.loop_func == 'z=f(x*z)':
            f_output = self._backbone(inputs_embeds=output * embeds)  # [B, 2n + 1, d]
        else:
            raise NotImplementedError
        return f_output

    def forward(self, xs, ys):
        zs = self._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]
        embeds = self._read_in(zs)  # [B, 2n, d_in + 1] -> [B, 2n, d]
        embeds = embeds[:, -self.n_last_tokens:, :]  # Оставляем только n последних токенов с предыдущего шага
        if self.loop_func in ['z=f(x+z)']:
            output = torch.zeros_like(embeds)  # also of shape [B, n_last_tokens, d]
        elif self.loop_func in ['z=f(x*z)']:
            output = torch.ones_like(embeds)  # also of shape [B, n_last_tokens, d]
        else:
            raise NotImplementedError("Currently we only support loop function z=f(x+z) or z=f(x*z).")

        pred_list = []
        for idx in range(self.n_loops):
            output = self.f(output, embeds)
            prediction = self._read_out(output)  # [B, 2n, d] -> [B, 2n, 1]
            if self._pred_type == 'regression':
                y = prediction[:, self.ind::self.freq, 0]
            elif self._pred_type == 'classification':
                y = prediction[:, self.ind::self.freq]
            else:
                raise NotImplementedError
            pred_list.append(y)
            if not self.print_flag:
                print(idx)
                self.print_flag = True

        return pred_list


class TransformerModelResidualN(TransformerModel):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, n_last_tokens=10,
                 num_positions_to_zero=10, loop_func='z=f(x+z)', pred_type='regression', n_loops=10):
        super(TransformerModelResidualN, self).__init__(n_dims, n_positions, n_embd, n_layer, n_head, n_last_tokens,
                                                        pred_type)
        self.loop_func = loop_func
        self.num_positions_to_zero = num_positions_to_zero
        self.n_loops = n_loops

    def f(self, output, embeds):
        if self.loop_func == 'z=f(x+z)':
            f_output = self._backbone(inputs_embeds=output + embeds)
        else:
            raise NotImplementedError
        return f_output

    def forward(self, xs, ys):
        zs = self._combine(xs, ys)
        readin = self._read_in(zs)
        embeds = readin[:, -self.n_last_tokens:, :]
        output = torch.zeros_like(embeds) if self.loop_func == 'z=f(x+z)' else torch.ones_like(embeds)
        pred_list = []

        for idx in range(self.n_loops):
            output = self.f(output, embeds)
            output = zero_out_positions(output, self.num_positions_to_zero)
            prediction = self._read_out(output)
            y = prediction[:, self.ind::self.freq, 0]
            pred_list.append(y)
        return pred_list