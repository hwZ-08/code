import torch


def mia_ov(model, in_loader, out_loader, N, threshold, device):
    model.eval()
    TP, TN, FP, FN = 0, 0, 0, 0
    for idx, (inputs, targets) in enumerate(in_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicts = torch.max(outputs.data, 1)
        corrects = predicts.eq(targets.data).cpu().sum()
        if (corrects >= threshold):
            TP += 1
        else: 
            FN += 1

    for idx, (inputs, targets) in enumerate(out_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicts = torch.max(outputs.data, 1)
        corrects = predicts.eq(targets.data).cpu().sum()
        if (corrects >= threshold):
            FP += 1
        else: 
            TN += 1

    precision = (TP + 1) / (TP + FP + 2)
    recall = (TP + 1) / (TP + FN + 2)
    acc = 100. * (TP + TN) / (TP + TN + FP + FN)
    print(f"\n[MIA OV] N = {N}, threshold = {threshold}")
    print(f"TP, TN, FP, FN = {TP, TN, FP, FN}")
    print(f"precision = {precision:.3f}, recall = {recall:.3f}, acc = {acc:.3f}%")