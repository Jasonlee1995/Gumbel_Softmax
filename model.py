import torch
import torch.nn as nn
import torch.nn.functional as F

    
def gumbel_softmax(logits, temperature, gpu_id=None, hard=False):
    if gpu_id:
        gumbel = torch.distributions.Gumbel(torch.zeros(logits.shape).cuda(gpu_id), torch.ones(logits.shape).cuda(gpu_id))
    else:
        gumbel = torch.distributions.Gumbel(torch.zeros(logits.shape), torch.ones(logits.shape))
        
    if hard:
        n_classes = logits.size(-1)
        z = torch.argmax(logits + gumbel.sample(), dim=-1)
        return F.one_hot(z, n_classes).float()
    else:
        return F.softmax((logits + gumbel.sample()) / temperature, dim=-1)
    

class Gumbel_Softmax_VAE(nn.Module):
    def __init__(self, img_dim, K, N, temperature, init_weights=True, gpu_id=None):
        super(Gumbel_Softmax_VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(img_dim, 512), nn.ReLU(), 
                                     nn.Linear(512, 256), nn.ReLU(), 
                                     nn.Linear(256, N*K))
        self.decoder = nn.Sequential(nn.Linear(N*K, 256), nn.ReLU(), 
                                     nn.Linear(256, 512), nn.ReLU(), 
                                     nn.Linear(512, img_dim), nn.Sigmoid())
        
        self.K, self.N, self.temperature = K, N, temperature
        self.gpu = gpu_id
        
        if init_weights: self._initialize_weights()
        if gpu_id: torch.cuda.set_device(self.gpu)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def sample(self, logits, temperature):
        if self.training: return gumbel_softmax(logits, temperature, self.gpu, hard=False)
        else: return gumbel_softmax(logits, temperature, self.gpu, hard=True)

    def forward(self, x):
        x = self.encoder(x)
        logits = F.softmax(x.view(-1, self.N, self.K), dim=-1)
        sample = self.sample(torch.log(logits+1e-10), self.temperature).view(-1, self.N*self.K)
        output = self.decoder(sample)
        return output, logits