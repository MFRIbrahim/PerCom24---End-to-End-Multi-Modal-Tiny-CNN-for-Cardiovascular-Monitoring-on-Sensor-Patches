import numpy as np
import torch
import torch.nn.functional as F
import model_quant as M
import utils
from pathlib import Path
from sklearn.metrics import auc
import copy
import argparse

DEVICE = "cpu"

def train(train_loader, model, optimizer, afftr_params, modality):
    model.train()
    loss_total = 0
    for x1, x2, y, _ in train_loader:
        x1 = x1.to(device=DEVICE)
        x2 = x2.to(device=DEVICE)
        y = y.to(device=DEVICE)
        if modality == "affine":
            x1 += torch.Tensor(afftr_params[0])
            x1 *= torch.Tensor(afftr_params[1])
            x2 += torch.Tensor(afftr_params[2])
            x2 *= torch.Tensor(afftr_params[3])
            x = torch.concat((x1, x2), 3)
        elif modality == "noaffine":
            x = torch.concat((x1, x2), 3)
        elif modality == "ecg":
            x = x1
        elif modality == "pcg":
            x = x2
        output = model(x)
        loss = F.cross_entropy(output, y, reduction='sum') / 9676
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.detach().item()
    return loss_total


def val(val_loader, model, threshold, afftr_params, modality):
    model.eval()
    loss_total = 0
    total_acc = 0
    with torch.no_grad():
        for x1, x2, y, record_number in val_loader:
            x1 = x1.to(device=DEVICE)
            x2 = x2.to(device=DEVICE)
            y = y.to(device=DEVICE)
            if modality == "affine":
                x1 += torch.Tensor(afftr_params[0])
                x1 *= torch.Tensor(afftr_params[1])
                x2 += torch.Tensor(afftr_params[2])
                x2 *= torch.Tensor(afftr_params[3])
                x = torch.concat((x1, x2), 3)
            elif modality == "noaffine":
                x = torch.concat((x1, x2), 3)
            elif modality == "ecg":
                x = x1
            elif modality == "pcg":
                x = x2
            output = model(x)
            loss = F.cross_entropy(output, y, reduction='sum') / 2451
            loss_total += loss.detach().item()
            softmax_output = F.softmax(output, dim=1)
            acc = utils.get_acc(softmax_output, y)
            total_acc += acc

    return loss_total, total_acc

def val_record(val_loader, model, threshold, afftr_params, modality):
    model.eval()
    results_array = []
    with torch.no_grad():
        for x1, x2, y, record_number in val_loader:
            x1 = x1.to(device=DEVICE)
            x2 = x2.to(device=DEVICE)
            y = y.to(device=DEVICE)
            if modality == "affine":
                x1 += torch.Tensor(afftr_params[0])
                x1 *= torch.Tensor(afftr_params[1])
                x2 += torch.Tensor(afftr_params[2])
                x2 *= torch.Tensor(afftr_params[3])
                x = torch.concat((x1, x2), 3)
            elif modality == "noaffine":
                x = torch.concat((x1, x2), 3)
            elif modality == "ecg":
                x = x1
            elif modality == "pcg":
                x = x2
            output = model(x)
            softmax_output = F.softmax(output, dim=1)
            record_number = record_number.to(device=DEVICE)
            results = utils.get_acc_recordwise(softmax_output, y, record_number, threshold)
            results_array.append(results.detach())

    return torch.cat(results_array, dim=0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile", help="model file of the model you want to train a quantized version of")
    args = parser.parse_args()
    name = args.modelfile
    net_path = Path(name)
    input_list = name.split(".")[0].split("_")
    modality = input_list[0]
    input_size = int(input_list[1])
    conf = (0., 1., -100., 0.01, input_size, modality)
    net = M.TinyCNN(conf).to(DEVICE)
    checkpoint = torch.load(net_path, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint["model_state_dict"], strict=False)
    M._replace_relu(net)
    net.eval()
    net.fuse_model(is_qat=True)
    net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    net.train()
    torch.quantization.prepare_qat(net, inplace=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)


    shift1 = 0
    scale1 = 0
    shift2 = 0
    scale2 = 0
    for key, value in checkpoint["model_state_dict"].items():
        if key == "shift1":
            shift1 = value
        if key == "scale1":
            scale1 = value
        if key == "shift2":
            shift2 = value
        if key == "scale2":
            scale2 = value

    afftr_params = (shift1, scale1, shift2, scale2)
    current_epoch = 0
    n_epochs = 400 + current_epoch

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
    print("old accuracy:", old_acc)
    for i in range(current_epoch, n_epochs):
        print(i+1, "/ 400")
        train_loss = train(train_loader, net, optimizer, afftr_params, modality)
        net.eval()
        net_quant = torch.quantization.convert(net)
        eval_loss, acc = val(val_loader, net_quant, 0.5, afftr_params, modality)
        print(train_loss, eval_loss, acc, old_acc)
        if acc > old_acc:
            net_best = copy.deepcopy(net_quant)
            old_acc = acc

    thresholds = val_record(val_loader, net_best, 0.5, afftr_params, modality)[:, 3]
    thresholds = torch.unique(thresholds, sorted=True)
    max_threshold = torch.max(thresholds).item()
    min_threshold = torch.min(thresholds).item()
    thresholds = np.linspace(min_threshold, max_threshold, 100)
    thresholds[-1] = thresholds[-1]+1
    scores = []
    roc_coordinates = []
    indx = 0
    for threshold in thresholds:
        print(indx)
        indx += 1
        results = val_record(val_loader, net_best, threshold, afftr_params, modality)
        metrics, coords = utils.get_metrics(results, threshold)
        scores.append(metrics)
        roc_coordinates.append(coords)

    scores.sort(key=lambda x: (x[0],x[3]), reverse=True)
    print(scores[0])
    roc_coordinates.sort(key=lambda x: (x[0], x[1]))
    x_coordinates = []
    y_coordinates = []
    for coord in roc_coordinates:
        x_coordinates.append(coord[0])
        y_coordinates.append(coord[1])
    
    area_under_curve = auc(x_coordinates, y_coordinates)
    print(area_under_curve)

if __name__ == "__main__":
    main()