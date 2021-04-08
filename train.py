import math
import model, torch
from torch import nn, optim
import torch.nn.functional as F


class Gumbel_Softmax_VAE():
    def __init__(self, img_dim=28*28, K=10, N=20, temperature=1, init_weights=True, gpu_id=0, print_freq=10, epoch_print=10):
        self.K, self.N = K, N
        self.gpu = gpu_id
        self.print_freq = print_freq
        self.epoch_print = epoch_print

        torch.cuda.set_device(self.gpu)
        
        self.model = model.Gumbel_Softmax_VAE(img_dim=img_dim, K=K, N=N, temperature=temperature, 
                                              init_weights=init_weights, gpu_id=gpu_id).cuda(self.gpu)

        self.train_losses = []
        self.test_losses = []

    def train(self, train_data, test_data, epochs, lr, weight_decay):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        
        for epoch in range(epochs):
            if epoch % self.epoch_print == 0: print('Epoch {} Started...'.format(epoch+1))
            for i, (X) in enumerate(train_data):
                X = X.cuda(self.gpu).float().view(X.size(0), -1)
                X_prob, logits = self.model(X)
                logits = logits.view(-1, self.N*self.K)

                reconstruction_loss = -F.binary_cross_entropy(X_prob, X, reduction='sum') / X.size(0)
                kl_loss = -(logits * torch.log(logits*self.K + 1e-10)).sum(dim=-1).mean()
                
                elbo = reconstruction_loss + kl_loss
                loss = -elbo

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % self.print_freq == 0:
                    test_loss = self.test(test_data)
                    self.train_losses.append(loss.item())
                    self.test_losses.append(test_loss)
                    if epoch % self.epoch_print == 0:
                        print('Iteration : {} - Train Loss : {:.4f}, Test Loss : {:.4f}'.format(i+1, loss.item(), test_loss))

    def test(self, test_data):
        losses, total = 0, 0

        self.model.eval()
        with torch.no_grad():
            for i, (X) in enumerate(test_data):
                total += X.size(0)
                X = X.cuda(self.gpu).float().view(X.size(0), -1)
                X_prob, logits = self.model(X)
                logits = logits.view(-1, self.N*self.K)

                reconstruction_loss = -F.binary_cross_entropy(X_prob, X, reduction='sum')
                kl_loss = -(logits * torch.log(logits*self.K + 1e-10)).sum(dim=-1).sum()
                
                elbo = reconstruction_loss + kl_loss
                loss = -elbo
                losses += loss
                
        self.model.train()
        return losses/total