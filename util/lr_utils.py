def backbone_head_lr(model, name, lr=1e-5, lr_scale=0.1):
    flag = False
    for n, p in model.named_parameters():
        if name in n:
            flag = True
    assert flag == True, f"Cannot find {name} in model parameters."

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if name not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() if name in n and p.requires_grad],
            "lr": lr * lr_scale,
            "lr_scale": lr_scale
        },
    ]

    return param_dicts