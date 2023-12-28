import torch.nn.functional as F
import torch
import scipy.stats


def evidence(outputs):
    evidence = F.softplus(outputs)
    alpha = evidence + 1
    S = torch.sum(alpha,dim=1,keepdim=True)
    b_k = evidence / S
    u_k = 1 - torch.sum(b_k,dim=1)
    prob = alpha / S
    y_with_uncertainty = torch.cat((b_k,u_k.unsqueeze(1)),dim=1)
    return  prob, y_with_uncertainty


def mask_generation_t(pseudo_label):
    mask_uc = ((pseudo_label == 4).float()).bool()
    mask_c = ((pseudo_label != 4).float()).bool()
    return mask_c, mask_uc

def mask_generation(pseudo_label,true_labels):
    mask_r = ((pseudo_label == true_labels).float()).bool()
    return mask_r

