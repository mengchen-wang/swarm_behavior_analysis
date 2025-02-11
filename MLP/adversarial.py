"""
code adapted from https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from time import time

def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def project(x: np.array, original_x: np.array, epsilon, _type='l2'):

    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon
        x = torch.max(torch.min(x, max_x), min_x)

    elif _type == 'l2':
        dist = (x - original_x)
        dist = dist.view(x.shape[0], -1)
        dist_norm = torch.norm(dist, dim=1, keepdim=True)
        print(dist_norm)
        mask = (dist_norm > epsilon)
        print(mask)
        dist = dist / dist_norm
        dist *= epsilon
        dist = dist.view(x.shape)
        x = (original_x + dist) * mask.float() + x * (1 - mask.float())

    else:
        raise NotImplementedError

    return x

class FastGradientSignUntargeted():
    def __init__(self, model, epsilon, alpha, max_iters, _type='linf'):
        self.model = model
        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type
        
    def perturb(self, original_images, labels, reduction4loss='mean', random_start=False):
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.to(device)
            x = original_images + rand_perturb
        else:
            x = original_images.clone()

        x.requires_grad = True 

        self.model.eval()

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                outputs = self.model(x)

                loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)

                if reduction4loss == 'none':
                    grad_outputs = tensor2cuda(torch.ones(loss.shape))
                    
                else:
                    grad_outputs = None

                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, 
                        only_inputs=True)[0]

                x.data += self.alpha * torch.sign(grads.data) 
                x = project(x, original_images, self.epsilon, self._type)

        self.model.train()

        return x


def train(model, max_epoch, n_eval_step, train_loader, weight_decay, learning_rate, epsilon, alpha, max_iters, _type, adv_train=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=[25, 50, 75], 
                                                         gamma=0.5)
    _iter = 0
    begin_time = time()
    for epoch in range(1, max_epoch+1):
        scheduler.step()
        for data, label in train_loader:
            data, label = data.to(device).to(torch.float32), label.to(device)
            if adv_train:
                attack = FastGradientSignUntargeted(model, epsilon, alpha, max_iters, _type)
                adv_data = attack.perturb(data, label, 'mean', True)
                output = model(adv_data)
            else:
                output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if _iter % n_eval_step == 0:
                if adv_train:
                    with torch.no_grad():
                        stand_output = model(data)
                    pred = torch.max(stand_output, dim=1)[1]
                    std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
                    pred = torch.max(output, dim=1)[1]
                    adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                else:
                    attack = FastGradientSignUntargeted(model, epsilon, alpha, max_iters, _type)
                    adv_data = attack.perturb(data, label, 'mean', False)
                    with torch.no_grad():
                        adv_output = model(adv_data)
                    pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100
                    pred = torch.max(output, dim=1)[1]
                    std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100                  

                print('epoch: %d, iter: %d, spent %.2f s, tr_loss: %.3f' % (
                    epoch, _iter, time()-begin_time, loss.item()))

                print('standard acc: %.3f %%, robustness acc: %.3f %%' % (
                    std_acc, adv_acc))
                begin_time = time()
            _iter += 1