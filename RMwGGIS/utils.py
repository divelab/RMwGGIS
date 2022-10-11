import numpy as np
from sympy.combinatorics.graycode import GrayCode, bin_to_gray, gray_to_bin
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# ### Code adapted https://github.com/google-research/google-research/tree/master/aloe/aloe
# def get_binmap(discrete_dim):
#     b = discrete_dim // 2 - 1
#     all_bins = []
#     for i in range(1 << b):
#         print(i/2**b)
#         bx = np.binary_repr(i, width=discrete_dim // 2 - 1)
#         all_bins.append('0' + bx)
#         all_bins.append('1' + bx)
#     vals = all_bins[:]
 
#     print('remapping binary repr with gray code')
#     a = GrayCode(b)
#     vals = []
#     for x in a.generate_gray():
#         vals.append('0' + x)
#         vals.append('1' + x)
#     bm = {}
#     inv_bm = {}
#     for i, key in enumerate(all_bins):
#         bm[key] = vals[i]
#         inv_bm[vals[i]] = key
#     return bm, inv_bm

# ### Code adapted https://github.com/google-research/google-research/tree/master/aloe/aloe
# def compress(x, discrete_dim):
#     bx = np.binary_repr(int(abs(x)), width=discrete_dim // 2 - 1)
#     bx = '0' + bx if x >= 0 else '1' + bx
#     return bx

# ### Code adapted https://github.com/google-research/google-research/tree/master/aloe/aloe
# def float2bin(samples, bm, discrete_dim, int_scale):
#     bin_list = []
#     for i in range(samples.shape[0]):
#         x, y = samples[i] * int_scale
#         bx, by = compress(x, discrete_dim), compress(y, discrete_dim)
#         bx, by = bm[bx], bm[by]
#         bin_list.append(np.array(list(bx + by), dtype=int))
#     return np.array(bin_list)


### Do not generate a map from binary code to gray code, which is time comsuming if data is high dimensional. Instead, we convert binary code to gray code on the fly.
def compress_new(x, discrete_dim):
    bx = np.binary_repr(int(abs(x)), width=discrete_dim // 2 - 1)
    return bx


def float2bin_new(samples, discrete_dim, int_scale):
    bin_list = []
    for i in range(samples.shape[0]):
        x, y = samples[i] * int_scale
        bx, by = compress_new(x, discrete_dim), compress_new(y, discrete_dim)
        bx, by = bin_to_gray(bx), bin_to_gray(by)
        bx = '0' + bx if x >= 0 else '1' + bx
        by = '0' + by if y >= 0 else '1' + by
        bin_list.append(np.array(list(bx + by), dtype=int))
    return np.array(bin_list)


def bin2float_new(samples, discrete_dim, int_scale):
    floats = []
    for i in range(samples.shape[0]):
        s = ''
        for j in range(samples.shape[1]):
            s += str(int(samples[i, j]))
        gx, gy = s[1:discrete_dim//2], s[discrete_dim//2+1:]
        x, y = int(gray_to_bin(gx), 2), int(gray_to_bin(gy), 2)
        x = x if s[0] == "0" else -x
        y = y if s[discrete_dim//2] == "0" else -y
        x /= int_scale
        y /= int_scale
        floats.append((x, y))
    return np.array(floats)


### Code adapted https://github.com/google-research/google-research/tree/master/aloe/aloe
# def plot_heat(model, bm, device, discrete_dim, int_scale, out_file=None):
def plot_heat(model, device, discrete_dim, int_scale, out_file=None):
    plt.figure(figsize=(4.1,4.1))
    w = 100
    size = 4.1
    x = np.linspace(-size, size, w)
    y = np.linspace(-size, size, w)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, [-1, 1])
    yy = np.reshape(yy, [-1, 1])
    heat_samples = float2bin_new(np.concatenate((xx, yy), axis=-1), discrete_dim, int_scale)
    heat_samples = torch.from_numpy(heat_samples).to(device).float()
    heat_score = F.softmax(-1 * model(heat_samples).view(1, -1), dim=-1)
    a = heat_score.view(w, w).data.cpu().numpy()
    a = np.flip(a, axis=0)
    plt.imshow(a)
    plt.axis('equal')
    plt.axis('off')
    if out_file is None:
        out_file = os.path.join(cmd_args.save_dir, 'heat.png')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    

### Code adapted https://github.com/google-research/google-research/tree/master/aloe/aloe
def plot_samples(samples, out_file, lim=4.1, axis=True):
    plt.figure(figsize=(4.1,4.1)) 
    plt.scatter(samples[:, 0], samples[:, 1], marker='.')
    plt.axis('equal')
    if lim is not None:
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
    if not axis:
        plt.axis('off')
    
    plt.xticks([])
    plt.yticks([])
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()

    
### Code adapted https://github.com/google-research/google-research/tree/master/aloe/aloe
def gibbs_step(orig_samples, axis, n_choices, model):
    orig_samples = orig_samples.clone()
    with torch.no_grad():
        cur_samples = orig_samples.clone().repeat(n_choices, 1)
        b = torch.LongTensor(list(range(n_choices))).to(cur_samples.device).view(-1, 1)
        b = b.repeat(1, orig_samples.shape[0]).view(-1)
        cur_samples[:, axis] = b
        score = model(cur_samples).view(n_choices, -1).transpose(0, 1)

        prob = F.softmax(-1 * score, dim=-1)
        samples = torch.multinomial(prob, 1)
        orig_samples[:, axis] = samples.view(-1)
    return orig_samples


### Code adapted https://github.com/google-research/google-research/tree/master/aloe/aloe
class GibbsSampler(nn.Module):
    def __init__(self, n_choices, discrete_dim, device):
        super(GibbsSampler, self).__init__()
        self.n_choices = n_choices
        self.discrete_dim = discrete_dim
        self.device = device

    def forward(self, model, num_rounds, num_samples=None, init_samples=None):
        assert num_samples is not None or init_samples is not None
        if init_samples is None:
            init_samples = torch.randint(self.n_choices, (num_samples, self.discrete_dim)).to(self.device)

        with torch.no_grad():
            cur_samples = init_samples.clone()
            for r in range(num_rounds):
                for i in range(self.discrete_dim):
                    cur_samples = gibbs_step(cur_samples, i, self.n_choices, model)

        return cur_samples


### Code adapted https://github.com/google-research/google-research/tree/master/aloe/aloe
def hamming_mmd(x, y):
    x = x.float()
    y = y.float()
    with torch.no_grad():
        kxx = torch.mm(x, x.transpose(0, 1))
        idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
        kxx = kxx * (1 - torch.eye(x.shape[0]).to(x.device))
        kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

        kyy = torch.mm(y, y.transpose(0, 1))
        idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
        kyy[idx, idx] = 0.0
        kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
        kxy = torch.sum(torch.mm(y, x.transpose(0, 1))) / x.shape[0] / y.shape[0]
        mmd = kxx + kyy - 2 * kxy
    return mmd


### Code adapted https://github.com/google-research/google-research/tree/master/aloe/aloe
def estimate_hamming_old(model, true_samples, rand_samples, gibbs_sampler, num_rounds_gibbs):
    with torch.no_grad():
        gibbs_samples = gibbs_sampler(model, num_rounds_gibbs, init_samples=rand_samples)
        return hamming_mmd(true_samples, gibbs_samples), gibbs_samples
    
    
def hamming_mmd_new(x, y):
    x = x.float()
    y = y.float()
    with torch.no_grad():
        kxx = x.shape[-1]-(x[:, None, :] != x).sum(2) ### Each element corresponds to #Dimension-HammingDistance
        idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
        kxx = kxx * (1 - torch.eye(x.shape[0]).to(x.device)).float()
        kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)
        
        

        kyy = y.shape[-1]-(y[:, None, :] != y).sum(2)
        idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
        kyy[idx, idx] = 0.0
        kyy = kyy.float()
        kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
        
        kxy = x.shape[-1]-(y[:, None, :] != x).sum(2).float()
        kxy = torch.sum(kxy) / x.shape[0] / y.shape[0]
        mmd = kxx + kyy - 2 * kxy
    return mmd
    
    
def estimate_hamming_new(model, true_samples, rand_samples, gibbs_sampler, num_rounds_gibbs):
    with torch.no_grad():
        gibbs_samples = gibbs_sampler(model, num_rounds_gibbs, init_samples=rand_samples)
        return hamming_mmd_new(true_samples, gibbs_samples), gibbs_samples