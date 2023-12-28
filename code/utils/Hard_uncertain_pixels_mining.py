import torch
from collections import defaultdict
import torch.nn.functional as F

def soft_multi_labelling(uncertain_features,memory):
    temp = 0.5
    similarity_list = []
    Mapping_dist = defaultdict(list)
    deMapping_list = defaultdict(list)
    index = 0
    for class_index in memory:
        Mapping_dist[index] = class_index
        deMapping_list[class_index] = index
        if len(memory[class_index]) == 1:
            proto = memory[class_index][0]
        else:
            proto = torch.cat((memory[class_index][0],memory[class_index][0]),dim = 0)

        prototypes = torch.tensor(proto).cuda()
        prototypes = F.normalize(prototypes, dim=1)
        uncertain_features_norm = F.normalize(uncertain_features, dim=1)
        similarities = torch.mm(uncertain_features_norm, prototypes.transpose(1, 0))
        socres = torch.exp(torch.mean(similarities, dim=1))
        similarity_list.append(socres.unsqueeze(socres.dim()))
        index += 1
    M_prediction = torch.cat(similarity_list, 1)
    m = torch.softmax(M_prediction / temp, dim = 1)
    return m, Mapping_dist, deMapping_list


def refinement(m_prob,hard_predcition_t,mapping_list,subset_prtotpye_memory,num_classes,mask_uc_t):
    _, subset_num = m_prob.shape
    R = torch.zeros(m_prob.shape[0],num_classes).cuda()
    for i in range(subset_num):
        keys = list(mapping_list[i][0])
        if(len(keys) == 1):
            index = int(keys[0])
            R[:,index] += m_prob[:,i]
        else:
            for c in keys:
                index = int(c)
                p_1 = subset_prtotpye_memory[str(c)]
                p_2 = subset_prtotpye_memory[str(keys)]
                inner_product = torch.sum(p_1*p_2, dim = 1)
                norm_1 = torch.norm(p_1,dim = 1)
                w_1 = (inner_product/norm_1).mean()
                R[:,index] += w_1 * m_prob[:,i]

    hard_predcition_t = hard_predcition_t.permute(0, 2, 3, 1)
    hard_predcition_t = hard_predcition_t[mask_uc_t]
    _, hp_index = torch.sort(hard_predcition_t,1,True)
    _, r = torch.max(R, dim = 1)

    indicator = torch.zeros(hard_predcition_t.shape[0])
    for index,it in enumerate(r):
        indicator[index] = (hp_index[index,:2].eq(int(it)).sum().bool())

    return indicator,r



