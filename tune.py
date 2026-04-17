from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import deepinv as dinv
from .utils import demosaic_gaussian, get_4x4_mask, RAM_init_hijack

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to dataset.", default=None)
    parser.add_argument("--epochs", type=int, help="Num. epochs for fine-tuning", default=100)
    parser.add_argument("--model_pth",type=str, default="", help="Pretrained model path.")
    parser.add_argument("--out", type=str, default="", help="Out directory.")
    args = parser.parse_args()

    device = dinv.utils.get_device()

    # Load data
    ### NOTE: replace with your own dataset. The dataset should return tuples (x, y) where 
    ### x is ground truth of shape C,H,W, or torch.nan if ground truth does not exist.
    ### y is measurements of shape C,H,W, where each channel's pixels are 0 if they are masked and nonzero if they are selected by the filter.
    ### Note the data will be batched by the dataloader.
    data = np.load(args.dataset)
    dataset = dinv.datasets.TensorDataset(
        x=torch.from_numpy(data["x"]), # shape N,C,H,W where N is total number of images
        y=torch.from_numpy(data["y"]), # shape N,C,H,W where N is total number of images
    )

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1], generator=torch.Generator("cpu").manual_seed(0))
    train_dataloader, val_dataloader, test_dataloader = torch.utils.data.DataLoader(train_dataset), torch.utils.data.DataLoader(val_dataset), torch.utils.data.DataLoader(test_dataset)
    x_test, y_test = next(iter(test_dataloader))
    y_test = y_test.to(device)
    
    # Get mask (multispectral filter array)
    ### NOTE: replace with your mask, where mask is of shape B,C,H,W such that y = mask * x.
    ### Here we assume the mask is sequential.
    mask = get_4x4_mask(y_test).to(device)
    
    physics = dinv.physics.Inpainting(mask.shape[1:], mask=mask, device=device, noise_model=dinv.physics.ZeroNoise())

    # Use multispectral RAM (https://arxiv.org/abs/2503.08915) with an initial reconstruction at model input.
    model = RAM_init_hijack(custom_init=demosaic_gaussian, pretrained=False, device=device, in_channels=(y_test.shape[1],))
    
    # Load model from state dict.
    model.load_state_dict(torch.load(args.model_pth, map_location=device, weights_only=True))

    # Test results for base model
    if torch.isnan(x_test).all(): # no GT
        metrics = [dinv.metric.Metric(metric=lambda *args, **kwargs: torch.tensor([0.]))]
    else:
        metrics = [dinv.metric.PSNR(), dinv.metric.SSIM(), dinv.metric.SpectralAngleMapper(), dinv.metric.ERGAS(factor=4)]

    tester_kwargs = {"metrics": metrics, "optimizer": None, "train_dataloader": None, "device": device, "save_path": None, "verbose": True, "show_progress_bar": True,}

    dinv.Trainer(model, physics, **tester_kwargs).test(test_dataloader)

    # Fine-tuning with proposed loss
    losses = [dinv.loss.MCLoss(metric=torch.nn.L1Loss()), dinv.loss.EILoss(dinv.transform.projective.PanTiltRotate(theta_z_max=0, theta_max=2, interpolation="bicubic", device=device), weight=0.1, metric=torch.nn.L1Loss())]

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        save_path=None,
        metrics=metrics,
        early_stop=5,
        device=device,
        losses=losses,
        epochs=args.epochs,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150]),
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        early_stop_on_losses=True,
        compute_eval_losses=True, # for early stopping when no GT
        compute_train_metrics=False,
    )
    trainer.train()
    model_tuned = trainer.model
    
    # Test tuned model
    dinv.Trainer(model_tuned, physics, **tester_kwargs).test(test_dataloader)

    # Visual results
    with torch.no_grad():
        x_lin = demosaic_gaussian(y_test, physics).detach().cpu()
        x_net = model_tuned(y_test, physics).detach().cpu()

    dinv.utils.plot({
        f"Interpolation {metrics[0](x_lin, x_test).item():.2f}": x_lin[:, [3, 11, 12]],
        f"Reconstruction {metrics[0](x_net, x_test).item():.2f}": x_net[:, [3, 11, 12]],
    } | (
        {} if x_test.isnan().all() else {"GT": x_test[:, [3, 11, 12]]}
    ), save_fn=Path(args.out) / ".png", fontsize=7)