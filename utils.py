import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import dataclass
from torchvision import transforms

TRANSFORMS_ECG_TRAIN_64 = torch.nn.Sequential(
    transforms.Normalize((1.696053804538871), (5.0027720495821235))
)
TRANSFORMS_PCG_TRAIN_64 = torch.nn.Sequential(
    transforms.Normalize((-0.8056607714470333), (234.7629578272776))
)

TRANSFORMS_ECG_TRAIN_128 = torch.nn.Sequential(
    transforms.Normalize((1.6960538001611747), (5.059049272430157))
)
TRANSFORMS_PCG_TRAIN_128 = torch.nn.Sequential(
    transforms.Normalize((-0.8056608109637546), (778.7793326182677))
)

TRANSFORMS_ECG_TRAIN_256 = torch.nn.Sequential(
    transforms.Normalize((1.6960538056303067), (5.116592606111553))
)
TRANSFORMS_PCG_TRAIN_256 = torch.nn.Sequential(
    transforms.Normalize((-0.8056608129442108), (1575.25380621162))
)

TRANSFORMS_ECG_TRAIN_6000 = torch.nn.Sequential(
    transforms.Normalize((1.6960538200208575), (5.17892202064549))
)
TRANSFORMS_PCG_TRAIN_6000 = torch.nn.Sequential(
    transforms.Normalize((-0.805660742062319), (1971.52893859227))
)

def get_train_loader(input_dir, input_dir2, input_size):
    if input_size == 64:
        transform = TRANSFORMS_ECG_TRAIN_64
        transform2 = TRANSFORMS_PCG_TRAIN_64
    elif input_size == 128:
        transform = TRANSFORMS_ECG_TRAIN_128
        transform2 = TRANSFORMS_PCG_TRAIN_128
    elif input_size == 256:
        transform = TRANSFORMS_ECG_TRAIN_256
        transform2 = TRANSFORMS_PCG_TRAIN_256
    elif input_size == 6000:
        transform = TRANSFORMS_ECG_TRAIN_6000
        transform2 = TRANSFORMS_PCG_TRAIN_6000
    dataset = dataclass.HybridDatasetnpy(input_dir, input_dir2, transform=transform, transform2=transform2)
    batch_size = 32
    targets = []
    for _, _, y, _ in dataset:
        if y[0] == 1:
            targets.append(0)
        else:
            targets.append(1)

    counts = [2281, 7395]
    weights = [1/counts[i] for i in targets]

    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)

    return loader

def get_val_loader(input_dir, input_dir2, input_size):
    if input_size == 64:
        transform = TRANSFORMS_ECG_TRAIN_64
        transform2 = TRANSFORMS_PCG_TRAIN_64
    elif input_size == 128:
        transform = TRANSFORMS_ECG_TRAIN_128
        transform2 = TRANSFORMS_PCG_TRAIN_128
    elif input_size == 256:
        transform = TRANSFORMS_ECG_TRAIN_256
        transform2 = TRANSFORMS_PCG_TRAIN_256
    elif input_size == 6000:
        transform = TRANSFORMS_ECG_TRAIN_6000
        transform2 = TRANSFORMS_PCG_TRAIN_6000
    dataset = dataclass.HybridDatasetnpy(input_dir, input_dir2, transform=transform, transform2=transform2)
    batch_size = 32
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    return loader


def get_acc(output, target):
    predictions = output[:, 1] >= 0.5
    labels = target[:, 1]

    correct = (predictions==labels).sum().item()

    return correct / 2451


def get_acc_recordwise(output, target, record_number, threshold):
    predictions = output[:, 1] >= threshold
    labels = target[:, 1]

    results = torch.stack([predictions, labels, record_number, output[:, 1]], dim=1)

    return results


def get_metrics(results, threshold):
    final_predictions = []
    labels = []
    records = []
    for number in results[:, 2]:
        if number not in records:
            records.append(number.item())
    
    for record_number in records:
        record_number_part = results[results[:, 2] == record_number]
        label = record_number_part[0, 1]
        vote_ratio = record_number_part[:, 0].sum() / record_number_part.shape[0]
        final_predictions.append(int(vote_ratio.item() >= 0.5))
        labels.append(label.item())
    
    final_predictions = torch.tensor(final_predictions)
    labels = torch.tensor(labels)
    
    p = (labels==1).sum().item()
    n = (labels==0).sum().item()
    tp = (final_predictions[labels==1]).sum().item()
    tn = ((final_predictions[labels==0]) == 0).sum().item()
    fp = n - tn
    fn = p - tp 
    
    accuracy = (final_predictions==labels).sum().item() / len(labels)
    sensitivity = tp/p
    specificity = tn/n
    f1_score = 2*tp / (2*tp + fp + fn)
    if (tp + fp) == 0:
        precision = 0
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    return (accuracy, sensitivity, specificity, f1_score, precision), (1-specificity, sensitivity)