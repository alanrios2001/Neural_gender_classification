import torch

# funcao de acuracia e f1
def acc_fn(y_true, y_pred):
    correct = torch.eq(y_pred,y_true).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

def F1_score(prob, label):
    prob = prob.bool()
    label = label.bool()
    epsilon = 1e-7
    TP = (prob & label).sum().float()
    TN = ((~prob) & (~label)).sum().float()
    FP = (prob & (~label)).sum().float()
    FN = ((~prob) & label).sum().float()
    #accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = torch.mean(TP / (TP + FP + epsilon))
    recall = torch.mean(TP / (TP + FN + epsilon))
    F2 = 2 * precision * recall / (precision + recall + epsilon)
    return F2
