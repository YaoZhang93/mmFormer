import math
import numpy as np
import torch
import torch.nn.functional as F

class LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, mode='poly'):
        self.mode = mode
        self.lr = base_lr
        self.num_epochs = num_epochs

    def __call__(self, optimizer, epoch):
        if self.mode == 'poly':
            now_lr = round(self.lr * np.power(1 - np.float32(epoch)/np.float32(self.num_epochs), 0.9), 8) 
        self._adjust_learning_rate(optimizer, now_lr)
        return now_lr

    def _adjust_learning_rate(self, optimizer, lr):
        optimizer.param_groups[0]['lr'] = lr

def get_temperature(epoch):
    if epoch <= 29:
        return 31 - (epoch+1)
    else:
        return 1

def get_params(model):
    ignore_id = list(map(id, model.module.decoder_all.abstraction1.fusion_conv.attention.parameters())) + \
            list(map(id, model.module.decoder_all.abstraction2.fusion_conv.attention.parameters())) + \
            list(map(id, model.module.decoder_all.abstraction3.fusion_conv.attention.parameters())) + \
            list(map(id, model.module.decoder_all.abstraction4.fusion_conv.attention.parameters()))
    print ('ignore_id', ignore_id)

    ignore_params = filter(lambda p: id(p) in ignore_id, model.parameters())
    base_params = filter(lambda p: id(p) not in ignore_id, model.parameters())

    return base_params, ignore_params

def record_loss(args, writer, mask1, loss_list, loss_name, step, mask_list, name_list, p_type):
    for i in range(mask1.size(0)):
        for j in range(15):
            if torch.equal(mask1[i].int(), mask_list[j].int()):
                for k in range(len(loss_list)):
                    writer.add_scalar(p_type[i] + '_' + name_list[j] + '_' + loss_name[k], loss_list[k][i].item(), global_step=step)

def Js_div(feat1, feat2, KLDivLoss):
    log_pq = ((p+q) / 2).log()
    return (KLDivLoss(log_pq, p) + KLDivLoss(log_pq, q))/2

    
def mutual_learning_loss(mutual_feat, mask, KLDivLoss):
    mutual_loss = torch.zeros(mask.size(0)).cuda()
    for i in range(mask.size(0)):
        K = torch.sum(mask[i])
        if K == 1:
            continue
        for j in range(4):
            feat = mutual_feat[j][:, mask[i], :, :, :, :]
            feat = F.softmax(feat, dim=2) 
            for k in range(K):
                for k1 in range(k+1, K):
                    mutual_loss[i] += Js_div(feat[:, k, :, :, :, :], feat[:, k1, :, :, :, :], KLDivLoss)
        mutual_loss[i] = mutual_loss[i] / (2 * K * (K-1))



class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()
    def __len__(self):
        return len(self.batch_sampler.sampler)
    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler
    def __iter__(self):
        while True:
            yield from iter(self.sampler)
            
            
