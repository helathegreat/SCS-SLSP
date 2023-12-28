import torch
from collections import defaultdict
import math

class FeatureMemory:

    def __init__(self, elements_per_class=32, n_classes=2):
        self.elements_per_class = elements_per_class
        self.subset_memory = defaultdict(list)
        self.memory = [None] * n_classes
        self.n_classes = n_classes
        self.ptrl = [0] * n_classes

    def add_features_from_sample_learned(self, model, features, class_labels,elements_per_class):
        """
        Updates the memory bank with some quality feature vectors per class
        Args:
            model: segmentation model containing the self-attention modules (contrastive_class_selectors)
            features: BxFxWxH feature maps containing the feature vectors for the contrastive (already applied the projection head)
            class_labels:   BxWxH  corresponding labels to the [features]
            batch_size: batch size

        Returns:
        """
        features = features.detach()
        class_labels = class_labels.detach().cpu().numpy()

        for c in range(self.n_classes):
            mask_c = class_labels == c
            selector = model.__getattr__('contrastive_class_selector_' + str(c))
            features_c = features[mask_c, :]

            if features_c.shape[0] > 0:
                if features_c.shape[0] > elements_per_class:
                    with torch.no_grad():
                        rank = selector(features_c)
                        rank = torch.sigmoid(rank)
                        _, indices = torch.sort(rank[:, 0], dim=0)
                        indices = indices.cpu().numpy()
                        features_c = features_c.cpu().numpy()
                        features_c = features_c[indices, :]
                        self.ptrl[c] = elements_per_class
                        new_features = features_c[:elements_per_class, :]
                else:
                    if self.memory[c] is not None:
                        memory_c = torch.from_numpy(self.memory[c]).cuda()
                        new_old_feature =  torch.cat((memory_c,features_c),dim = 0)
                        if new_old_feature.shape[0] >= elements_per_class:
                            new_features = new_old_feature[-elements_per_class:, :]
                            new_features = new_features.cpu().numpy()
                            self.ptrl[c] = elements_per_class
                        else:
                            self.ptrl[c] = (self.ptrl[c] + features_c.shape[0]) % elements_per_class
                            new_features = new_old_feature.cpu().numpy()
                    else:
                        new_features = features_c.cpu().numpy()

                self.memory[c] = new_features

    def thres_cal(self,matching_set, mismatching_set):
        mean_pos = matching_set.mean().cuda()
        mean_neg = mismatching_set.mean().cuda()
        stnd_pos = matching_set.std()
        stnd_neg = mismatching_set.std()

        A = stnd_pos.pow(2) - stnd_neg.pow(2)
        B = 2 * ((mean_pos * stnd_neg.pow(2)) - (mean_neg * stnd_pos.pow(2)))
        C = (mean_neg * stnd_pos).pow(2) - (mean_pos * stnd_neg).pow(2) + 2 * (stnd_pos * stnd_neg).pow(2) * torch.log(
            stnd_neg / (0.01 * stnd_pos) + 1e-8)
        E = B.pow(2) - 4 * A * C

        if E > 0:
            thres = ((-B + torch.sqrt(E)) / (2 * A + 1e-10)).item()
        else:
            thres = (mean_pos + mean_neg)/2

        return thres, mean_pos, mean_neg

    def subset_prototype_generation(self,features,class_labels):
        features = features.detach()
        class_labels = class_labels.detach().cpu().numpy()

        feature_dict = defaultdict(list)

        for i in range(self.n_classes):
            feature_dict[str(i)].append(features[class_labels == i, :])


        similarity_dict = defaultdict(list)
        for c in range(self.n_classes):
            features_c = feature_dict[str(c)][0]
            if len(features_c) == 0:
                continue
            for s in range(self.n_classes):
                index = ''.join(sorted(str(c) + str(s)))
                memory_s = torch.from_numpy(self.memory[s]).cuda()
                if self.memory[s] is None:
                    continue
                dot_product = torch.mm(features_c, memory_s.transpose(1, 0))
                norm_a = torch.norm(features_c, dim=1, keepdim=True)
                norm_b = torch.norm(memory_s, dim=1, keepdim=True)
                similarities = dot_product / (norm_a * norm_b.t())
                similarity_dict[index].append(similarities.view(-1))

        for c in range(self.n_classes):
            for s in range(self.n_classes):
                if(c == s):
                    continue
                else:
                    index = ''.join(sorted(str(c) + str(s)))
                    if(len(similarity_dict[str(c) + str(c)]) == 0):
                        continue
                    matching_set = torch.tensor(similarity_dict[str(c) + str(c)][0])
                    mismatching_set = torch.tensor(similarity_dict[index][0]).clone().detach()
                    thres, mean_pos, mean_neg = self.thres_cal(matching_set,mismatching_set)
                    norm_neg = torch.norm(thres - mean_neg, p=2)
                    norm_pos = torch.norm(mean_pos - thres, p=2)
                    norm_overall = torch.norm(mean_pos - mean_neg,p=2)
                    w_1 = norm_neg/norm_overall
                    w_2  =norm_pos/norm_overall
                    self.subset_memory[index].append(w_1 * self.memory[c] + w_2 * self.memory[s])

    def merge(self):
        for c in range(self.n_classes):
            memory_c = torch.from_numpy(self.memory[c]).cuda()
            self.subset_memory[str(c)].append(memory_c)