import torch
import torch.nn.functional as F
import model as M
import utils
from pathlib import Path
import argparse


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(train_loader, model, optimizer):
    model.train()
    loss_total = 0
    for x1, x2, y, _ in train_loader:
        x1 = x1.to(device=DEVICE)
        x2 = x2.to(device=DEVICE)
        y = y.to(device=DEVICE)
        x = torch.concat((x1, x2), 3)
        output = model(x)
        loss = F.cross_entropy(output, y, reduction='sum') / 9676
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.detach().item()
    return loss_total


def val(val_loader, model, threshold=0.5):
    model.eval()
    loss_total = 0
    total_acc = 0
    with torch.no_grad():
        for x1, x2, y, record_number in val_loader:
            x1 = x1.to(device=DEVICE)
            x2 = x2.to(device=DEVICE)
            y = y.to(device=DEVICE)
            x = torch.concat((x1, x2), 3)
            output = model(x)
            loss = F.cross_entropy(output, y, reduction='sum') / 2451
            loss_total += loss.detach().item()
            softmax_output = F.softmax(output, dim=1)
            acc = utils.get_acc(softmax_output, y)
            total_acc += acc

    return loss_total, total_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model config you want to use for training: [affine OR noaffine OR ecg OR pcg]_[64 OR 128 OR 256 OR 6000]")
    args = parser.parse_args()
    name = args.model
    input_list = name.split("_")
    modality = input_list[0]
    input_size = int(input_list[1])
    conf = (0., 1., -100., 0.01, input_size, modality)
    net = M.TinyCNN(conf).to(DEVICE)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    current_epoch = 0
    n_epochs = 300 + current_epoch

    if input_size == 6000:
        path = "ecg_windows_6000_train"
        path2 = "pcg_windows_6000_train"
        path3 = "ecg_windows_6000_test"
        path4 = "pcg_windows_6000_test"
    
    elif input_size == 256:
        path = "ecg_windows_256_train"
        path2 = "pcg_windows_256_train"
        path3 = "ecg_windows_256_test"
        path4 = "pcg_windows_256_test"
    
    elif input_size == 128:
        path = "ecg_windows_128_train"
        path2 = "pcg_windows_128_train"
        path3 = "ecg_windows_128_test"
        path4 = "pcg_windows_128_test"
    
    elif input_size == 64:
        path = "ecg_windows_64_train"
        path2 = "pcg_windows_64_train"
        path3 = "ecg_windows_64_test"
        path4 = "pcg_windows_64_test"
    
    train_loader = utils.get_train_loader(path, path2, input_size)
    val_loader = utils.get_val_loader(path3, path4, input_size)
    
    old_acc = -1
    for i in range(current_epoch, n_epochs):
        print(i)
        train_loss = train(train_loader, net, optimizer)
        eval_loss, acc = val(val_loader, net)
        print(train_loss, eval_loss, acc, old_acc)
        if acc > old_acc:
            torch.save({
                'epoch': i,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': acc,
            }, f"{modality}_{input_size}_artifact_test.pth")
            old_acc = acc


if __name__ == "__main__":
    main()