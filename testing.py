import numpy as np
import torch
import torch.nn.functional as F
import model as M
import utils
from pathlib import Path
from sklearn.metrics import auc
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def val(val_loader, model, threshold=0.5):
    model.eval()
    results_array = []
    with torch.no_grad():
        for x1, x2, y, record_number in val_loader:
            x1 = x1.to(device=DEVICE)
            x2 = x2.to(device=DEVICE)
            y = y.to(device=DEVICE)
            x = torch.concat((x1, x2), 3)
            output = model(x)
            softmax_output = F.softmax(output, dim=1)
            record_number = record_number.to(device=DEVICE)
            results = utils.get_acc_recordwise(softmax_output, y, record_number, threshold)
            results_array.append(results.detach())

    return torch.cat(results_array, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modelfile", help="model file of the model you want to test")
    args = parser.parse_args()
    name = args.modelfile
    net_path = Path(name)
    input_list = name.split(".")[0].split("_")
    modality = input_list[0]
    input_size = int(input_list[1])
    conf = (0., 1., -100., 0.01, input_size, modality)
    net = M.TinyCNN(conf).to(DEVICE)
    checkpoint = torch.load(net_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    if input_size == 6000:
        path3 = "ecg_windows_6000_test"
        path4 = "pcg_windows_6000_test"
    
    elif input_size == 256:
        path3 = "ecg_windows_256_test"
        path4 = "pcg_windows_256_test"
    
    elif input_size == 128:
        path3 = "ecg_windows_128_test"
        path4 = "pcg_windows_128_test"
    
    elif input_size == 64:
        path3 = "ecg_windows_64_test"
        path4 = "pcg_windows_64_test"

    val_loader = utils.get_val_loader(path3, path4, input_size)
    
    thresholds = val(val_loader, net, 0.5)[:, 3]
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
        results = val(val_loader, net, threshold)
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