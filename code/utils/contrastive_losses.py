"""
More details can be checked at https://github.com/Shathe/SemiSeg-Contrastive
Thanks the authors for providing such a model to achieve the class-level separation.
"""
import torch
import torch.nn.functional as F


def labeled_contra_loss(model, features, class_labels, num_classes, memory):
    """
    Args:
        model: segmentation model that contains the self-attention MLPs for selecting the features
        to take part in the contrastive learning optimization
        features: Nx256  feature vectors for the contrastive learning (after applying the projection and prediction head)
        class_labels: N corresponding class labels for every feature vector
        num_classes: number of classesin the dataet
        memory: memory bank [List]

    Returns:
        returns the contrastive loss between features vectors from [features] and from [memory] in a class-wise fashion.
    """
    loss = 0
    for c in range(num_classes):
        if(c == 0):
            continue
        mask_c = class_labels == c
        features_c = features[mask_c,:]
        memory_c = memory[c]

        selector = model.__getattr__('contrastive_class_selector_' + str(c))
        selector_memory = model.__getattr__('contrastive_class_selector_memory' + str(c))

        if memory_c is not None and features_c.shape[0] > 1 and memory_c.shape[0] > 1:

            memory_c = torch.from_numpy(memory_c).cuda()

            memory_c = F.normalize(memory_c, dim=1)
            features_c_norm = F.normalize(features_c, dim=1)

            similarities = torch.mm(features_c_norm, memory_c.transpose(1, 0))
            distances = 1 - similarities

            learned_weights_features = selector(features_c.detach())
            learned_weights_features_memory = selector_memory(memory_c)

            learned_weights_features = torch.sigmoid(learned_weights_features)
            rescaled_weights = (learned_weights_features.shape[0] / learned_weights_features.sum(dim=0)) * learned_weights_features #[179013,1]
            rescaled_weights = rescaled_weights.repeat(1, distances.shape[1])
            distances = distances * rescaled_weights

            learned_weights_features_memory = torch.sigmoid(learned_weights_features_memory)
            learned_weights_features_memory = learned_weights_features_memory.permute(1, 0)
            rescaled_weights_memory = (learned_weights_features_memory.shape[0] / learned_weights_features_memory.sum(dim=0)) * learned_weights_features_memory
            rescaled_weights_memory = rescaled_weights_memory.repeat(distances.shape[0], 1)
            distances = distances * rescaled_weights_memory

            loss = loss + distances.mean()

    return loss / num_classes



def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):
    keys = keys.detach().clone().cpu()
    batch_size = keys.shape[0]
    ptr = int(queue_ptr)
    queue[0] = torch.cat((queue[0], keys.cpu()), dim=0)
    if queue[0].shape[0] >= queue_size:
        queue[0] = queue[0][-queue_size:, :]
        ptr = queue_size
    else:
        ptr = (ptr + batch_size) % queue_size
    queue_ptr[0] = ptr
    return batch_size


def unlabeled_contra_loss(
    rep,
    rep_teacher,
    pseudo_label_valid,
    demapping_list,
    memory,
    num_classes,
    memobank,
    queue_prtlis,
    queue_size,
    certain_mask_overall,
    soft_multi_labels,
    emmbeing_uncertain,
    momentum_prototype=None
):
    ref_dict = {"0": ["1", "2", "3", "12", "13", "23"], "1": ["2","3","23"], "2": ["1", "3", "13", "23"], "3": ["1","2","12"]}

    num_queries = 2000
    num_negatives = 100
    num_feat = 32
    temp = 0.5

    seg_certain = []
    seg_num_list = []
    seg_proto_list = []
    valid_classes = []
    new_keys = []

    rep_teacher = rep_teacher.permute(0, 2, 3, 1)
    all_valid = rep_teacher[certain_mask_overall,...]
    valid_label = pseudo_label_valid[certain_mask_overall]


    for i in range(num_classes):
        negative_class = ref_dict[str(i)]

        mask = (valid_label == i).float().bool()
        seg_certain.append(all_valid[mask,...])


        negative_samples = []
        for idx in negative_class:
            if len(idx) == 1:
                mask_not = (valid_label != int(idx)).float().bool()
                negative_samples.append(all_valid[mask_not,...])
            else:
                multi_label_pos = int(demapping_list[idx][0])
                mm = soft_multi_labels == multi_label_pos
                negative_samples.append(emmbeing_uncertain[mm,...])
                print("start soft-multi-labels")

        keys = torch.cat(negative_samples,dim=0)

        new_keys.append(
            dequeue_and_enqueue(
                keys=keys,
                queue=memobank[i],
                queue_ptr=queue_prtlis[i],
                queue_size=queue_size[i],


            )
        )

        memory_c = memory[i]
        memory_c = torch.from_numpy(memory_c).cpu()
        mu = torch.mean(memory_c, dim=0)
        seg_proto_list.append(mu.unsqueeze(0))

        if mask.sum() > 0:
            seg_num_list.append(int(mask.sum().item()))
            valid_classes.append(i)


    if (len(seg_num_list) <= 1):

        if momentum_prototype is None:
            return new_keys, torch.tensor(0.0) * rep.sum()
        else:
            return momentum_prototype, new_keys, torch.tensor(0.0) * rep.sum()
    else:
        reco_loss = torch.tensor(0.0).cuda()

        seg_proto = torch.cat(seg_proto_list)
        valid_seg = len(seg_num_list)

        for i in range(valid_seg):
            if (len(seg_certain[i]) > 0 and memobank[valid_classes[i]][0].shape[0] > 0):
                seg_low_entropy_idx = torch.randint(len(seg_certain[i]),size=(num_queries,))
                anchor_feat = (seg_certain[i][seg_low_entropy_idx].clone().cuda())
            else:

                reco_loss = reco_loss + 0 * rep.sum()
                continue

            with torch.no_grad():
                negative_feat = memobank[valid_classes[i]][0].clone().cuda()

                negative_idx = torch.randint(len(negative_feat), size=(num_queries * num_negatives,))

                negative_feat = negative_feat[negative_idx]
                negative_feat = negative_feat.reshape(num_queries, num_negatives, num_feat)

                positive_feat = (
                    seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1).cuda()
                )

                all_feat = torch.cat(
                    (positive_feat, negative_feat), dim=1
                )

            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)

            reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda())

        if momentum_prototype is None:
            return new_keys, reco_loss / valid_seg

if __name__ == '__main__':
    print("ok")

