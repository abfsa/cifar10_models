import os
import argparse
import yaml
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from models.resnet import *
from models.lenet import *


MODEL_REGISTRY = {
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'myresnet18': my_ResNet18,
    'myresnet34': my_ResNet34,
    'lenet': LeNet,
    'lenetresdropout': LeNetWithDropout,
    


}


def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup, mixup_alpha):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        if use_mixup:
            inputs, y_a, y_b, lam = mixup_data(inputs, targets, mixup_alpha, device)
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            acc = lam * get_accuracy(outputs, y_a) + (1 - lam) * get_accuracy(outputs, y_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = get_accuracy(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        accs.update(acc, inputs.size(0))

    return losses.avg, accs.avg

def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    accs   = AverageMeter()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = get_accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accs.update(acc, inputs.size(0))
    return losses.avg, accs.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CIFAR-10 with config file')

    parser.add_argument('--config', type=str, required=True,
                        help='path to YAML config file')
    parser.add_argument('--model-name', type=str, default= None)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if args.model_name:
        if isinstance(cfg.get('model'), dict):
            cfg['model']['name'] = args.model_name
        else:
            cfg['model'] = {'name': args.model_name, 'params': {}}

    save_dir = os.path.join(cfg['save_dir'], cfg['model']['name'])
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if cfg.get('gpu', False) and torch.cuda.is_available()
                          else 'cpu')

    model_cfg = cfg['model']
    model_name = model_cfg['name']
    assert model_name in MODEL_REGISTRY, f"Unknown model: {model_name}"
    model_params = model_cfg.get('params', {})
    model = MODEL_REGISTRY[model_name](**model_params).to(device)





    # read training parameters from the config file
    criterion = nn.CrossEntropyLoss() #交叉熵损失
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg['lr'],
                          momentum=cfg['momentum'],
                          weight_decay=cfg['weight_decay'])
    sched_cfg = cfg.get('scheduler', {})
    if sched_cfg.get('type') == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get('step_size', 30),
            gamma=sched_cfg.get('gamma', 0.1)
        )
    else:
        scheduler = None

    # trainloader
    aug = cfg.get('augmentation', {})
    train_loader, val_loader, _ = get_dataloaders(
        cfg['data_dir'], cfg['batch_size'],
        use_cutout=aug.get('cutout', False),
        cutout_length=aug.get('cutout_length', 16)
    )

    # logger save dir
    best_acc = 0.0
    metrics_csv = os.path.join(save_dir, 'metrics.csv')
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','train_acc','val_loss','val_acc'])

    # training loop
    for epoch in range(1, cfg['epochs'] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=aug.get('mixup', False),
            mixup_alpha=aug.get('mixup_alpha', 1.0)
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"[{epoch}/{cfg['epochs']}] "
              f"Train: loss={train_loss:.4f}, acc={train_acc:.2f}% | "
              f" Val: loss={val_loss:.4f}, acc={val_acc:.2f}%")

        # record into CSV
        with open(metrics_csv, 'a', newline='') as f:
            csv.writer(f).writerow(
                [epoch, train_loss, train_acc, val_loss, val_acc]
            )

        # save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f"Epoch {epoch}: New best model (val_acc={best_acc:.2f}%) saved to {best_path}")

        if scheduler:
            scheduler.step()

    # final model
    final_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Training finished! Best val acc: {best_acc:.2f}%. Final model: {final_path}")
























