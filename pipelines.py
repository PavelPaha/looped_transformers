import torch
from tqdm import tqdm

from config import build_model
from data import get_data


def train_step(args, model, xs, ys, optimizer, ctx, scaler):
    if args.model.family in ['gpt2', 'gpt2_tying', 'gpt2_cyclic']:
        if ctx is not None:
            with ctx:
                y_pred = model(xs, ys)
                ys = ys.squeeze(-1)
                loss = (ys - y_pred).square().mean()
        else:
            y_pred = model(xs, ys)
            loss = (ys - y_pred).square().mean()
    elif args.model.family in ['gpt2_loop', 'gpt2_residual_n']:
        if ctx is not None:
            with ctx:
                y_pred_list = model(xs, ys)
                y_pred_arr = torch.cat(y_pred_list, dim=0)
                y_star_arr = torch.cat([ys] * len(y_pred_list), dim=0).squeeze(-1)
                if y_star_arr.shape != y_pred_arr.shape:
                    min_len = min(y_star_arr.shape[1], y_pred_arr.shape[1])
                    y_star_arr = y_star_arr[:, :min_len]
                    y_pred_arr = y_pred_arr[:, :min_len]
                loss = (y_star_arr - y_pred_arr).square().mean()
                y_pred = y_pred_list[-1]
        else:
            y_pred_list = model(xs, ys)
            y_pred_arr = torch.cat(y_pred_list, dim=0)
            y_star_arr = torch.cat([ys] * len(y_pred_list), dim=0)
            if y_star_arr.shape != y_pred_arr.shape:
                min_len = min(y_star_arr.shape[1], y_pred_arr.shape[1])
                y_star_arr = y_star_arr[:, :min_len]
                y_pred_arr = y_pred_arr[:, :min_len]
            loss = (y_star_arr - y_pred_arr).square().mean()
            y_pred = y_pred_list[-1]

    if args.training.use_ctx:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    return loss.detach(), y_pred.detach()


def evaluate(model, dataloader, device, use_looped_output=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xs, ys in dataloader:
            xs, ys = xs.to(device), ys.to(device)
            y_pred = model(xs, ys)
            if use_looped_output:
                y_pred = y_pred[-1]

            ys = ys.squeeze(-1)
            y_pred = y_pred.squeeze(-1)

            if ys.shape != y_pred.shape:
                min_len = min(ys.shape[1], y_pred.shape[1])
                ys = ys[:, :min_len]
                y_pred = y_pred[:, :min_len]

            loss = (ys - y_pred).square().mean()
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def start_training(args, device):
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    if args.training.use_ctx:
        ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
        scaler = torch.cuda.amp.GradScaler()
    else:
        ctx = None
        scaler = None

    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.lr)

    train_loader = get_data(batch_size=32, data_size=1000)
    val_loader = get_data(batch_size=32, data_size=200)

    # Training loop
    losses = []
    gradients = []
    weight_norms = []
    gradient_norms = []

    avg_losses = []
    val_scores = []

    for epoch in tqdm(range(args.training.epochs)):
        model.train()
        total_loss = 0
        for batch_idx, (xs, ys) in enumerate(train_loader):
            xs, ys = xs.to(device), ys.to(device)
            loss, y_pred = train_step(args, model, xs, ys, optimizer, ctx, scaler)
            total_loss += loss.item()
            losses.append(loss.item())

            grads = []
            grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.cpu().clone())
                    grad_norm += param.grad.norm().item()
                    gradient_norms.append(grad_norm)

            gradients.append(grads)

        weight_norm = 0.0
        for param in model.parameters():
            weight_norm += param.norm().item()
        weight_norms.append(weight_norm)

        avg_losses.append(total_loss / len(train_loader))

        val_loss = evaluate(model, val_loader, device,
                            use_looped_output=args.model.family in ['gpt2_loop', 'gpt2_residual_n'])
        val_scores.append(val_loss)

    torch.save({
        'losses': losses,
        'avg_losses': avg_losses,
        'val_scores': val_scores,
        'gradients': gradients,
        'gradient_norms': gradient_norms,
        'weight_norms': weight_norms
    }, f'training_info_{args.name}.pt')


def compute_score(y_true, y_pred):
    y_true = y_true.squeeze(-1)
    return ((y_true - y_pred) ** 2).mean().item()
