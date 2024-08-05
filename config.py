from transformers import TransformerModel, TransformerModelLooped, TransformerModelTying, TransformerModelResidualN, \
    CyclicBlockTransformer


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            pred_type=conf.pred_type,
        )
    elif conf.family == 'gpt2_loop':
        model = TransformerModelLooped(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            pred_type=conf.pred_type,
            n_loops=conf.n_loop_window
        )
    elif conf.family == 'gpt2_cutted':
        model = TransformerModelResidualN(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            n_loops=conf.n_loop_window
        )
    elif conf.family == 'gpt2_cyclic':
        model = CyclicBlockTransformer(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            block_size=conf.block_size,
            pred_type=conf.pred_type,
            n_loops=conf.n_loop_window
        )
    else:
        raise NotImplementedError

    return model


class Args:
    def __init__(self, name, family, n_dims=16, n_positions=128, n_embd=64, n_layer=24, n_head=4, n_tokens=10,
                 n_loop_window=10, lr=0.0001, epochs=50, block_size=3, use_ctx=True, add_inputs_embeds=False,
                 train_data_size=1000,
                 val_data_size=200):
        self.name = name
        self.model = type('ModelArgs', (), {
            'family': family,
            'n_dims': n_dims,
            'n_positions': n_positions,
            'n_embd': n_embd,
            'n_layer': n_layer,
            'n_head': n_head,
            'pred_type': 'regression',
            'n_tokens': n_tokens,
            'n_loop_window': n_loop_window,
            'block_size': block_size
        })
        self.training = type('TrainingArgs', (), {
            'lr': lr,
            'epochs': epochs,
            'use_ctx': use_ctx,
            'add_inputs_embeds': add_inputs_embeds,
            'train_data_size': train_data_size,
            'val_data_size': val_data_size
        })