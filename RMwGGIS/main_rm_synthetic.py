import os
import sys
import numpy as np
import random
import torch
from torch.optim import Adam
import torch.autograd as autograd
import argparse
from texttable import Texttable
from tqdm import tqdm
from distutils.util import strtobool
import time
import json

from utils import plot_heat, plot_samples, estimate_hamming_new, GibbsSampler, float2bin_new, bin2float_new
from dataset import OnlineToyDataset
from model import EBM_MLP



### Args
parser = argparse.ArgumentParser()

### Setup
parser.add_argument('--seed', type=int, default=1023, help='Random seed to use')
parser.add_argument('--data_name', type=str, default='2spirals', help='Dataset name')
parser.add_argument('--discrete_dim', type=int, default=32, help='Data dimension')
parser.add_argument('--save_dir', type=str, default='./trained_models/', help='Location for saving checkpoints')
parser.add_argument('--save_interval', type=int, default=50, help='Interval (# of epochs) between saved checkpoints')
parser.add_argument('--plot', type=strtobool, default='false', help='Plot heat maps')
parser.add_argument('--plot_data', type=strtobool, default='false', help='Data distribution visualization')
parser.add_argument('--mmd', type=strtobool, default='true', help='Compute mmd')
parser.add_argument('--num_rounds_gibbs', type=int, default=20, help='Number of rounds of gibbs sampling')
parser.add_argument('--mmd_samples', type=int, default=4000, help='Number of used samples for computing MMD')
parser.add_argument('--record_name', type=str, default='None', help='Location for saving records')

### Energy function model
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
parser.add_argument('--bn', type=strtobool, default='false', help='Batch Normalization')
parser.add_argument('--activation', type=str, default='swish', help='Activation function')
# parser.add_argument('--std', type=float, default=1e-2, help='Std for spectral norm')

### Training hyperparameter
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--wd', type=float, default=0, help='Weight Decay')
parser.add_argument('--grad_clip', type=int, default=5, help='Clip gradient')
parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum training epochs')
parser.add_argument('--iter_per_epochs', type=int, default=100, help='Training iterations per epoch')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training')
parser.add_argument('--trained_model', type=str, default=None, help='Trained model')

args = parser.parse_args()

def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.set_precision(10)
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())


# def train(model, db, bm, inv_bm, device):
def train(model, db, device):
    if args.record_name is not None:
        train_records = {'energies':[], 'loss':[], 'mmd':[]}
    if args.trained_model is not None:
        model.load_state_dict(torch.load(args.trained_model))
        print('==========================================')
        print("Loading trained paramaters from {}".format(args.trained_model))
        ### Compute mmd 
        if args.mmd:  
            true_samples = float2bin_new(db.gen_batch(args.mmd_samples), args.discrete_dim, db.int_scale)
            true_samples = torch.from_numpy(true_samples).to(device).float()
            rand_samples = torch.randint(2, (args.mmd_samples, args.discrete_dim)).to(device).float()
            gibbs_sampler = GibbsSampler(2, args.discrete_dim, device)
            mmd, gen_samples = estimate_hamming_new(model, true_samples, rand_samples, gibbs_sampler, args.num_rounds_gibbs)
            print('MMD score based on the trained model: %.5f' % mmd)
            print('==========================================')
            plot_samples(bin2float_new(gen_samples.cpu().detach().numpy(), args.discrete_dim, db.int_scale), out_file=os.path.join(args.save_dir, '{}-gen_samples.png'.format(args.data_name)))
            print('Visualization generated samples')
            print('==========================================')
            sys.exit()
    
    rand_samples = torch.randint(2, (1000, args.discrete_dim)).to(device).float()
    print('==========================================')
    print('Energy (random samples) [before training]: %.4f±%.4f' % (torch.mean(model(rand_samples)).item(), torch.std(model(rand_samples)).item()))

    
    samples = float2bin_new(db.gen_batch(1000), args.discrete_dim, db.int_scale)
    samples = torch.from_numpy(samples).to(device).float()
    print('==========================================')
    print('Energy (positive samples) [before training]: %.4f±%.4f' % (torch.mean(model(samples)).item(), torch.std(model(samples)).item()))

    
    
    parameters = model.parameters()
    optimizer = Adam(parameters, lr=args.lr, weight_decay=args.wd)
    
    if args.plot:
        plot_heat(model, device, args.discrete_dim, db.int_scale, out_file=os.path.join(args.save_dir, 'heat-{}-init.png'.format(args.data_name)))
        print('Saving heat map')
        print('==========================================')
    

    for epoch in range(args.max_epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        pbar = tqdm(range(args.iter_per_epochs))
        losses = []
        for i in pbar:
            samples = float2bin_new(db.gen_batch(args.batch_size), args.discrete_dim, db.int_scale)
            samples = torch.from_numpy(samples).to(device).float()
            
            ### Used for Verification
#             samples.requires_grad_(True)
            

            
            
            energy_x = model(samples) ### [128, 1]
            
            
            ### Better implementation
            flip_samples = []
            for d in range(samples.shape[-1]):
                flip_samples_d = samples.clone()
                flip_samples_d[:,d] = 1 - flip_samples_d[:,d]
                flip_samples.append(flip_samples_d)
                
            flip_samples_cat = torch.cat(flip_samples, dim=0) ### [128*32, 32]
            energy_xp = torch.reshape(model(flip_samples_cat), (-1, samples.shape[-1])) ### [128, 32]
            
            ### For evaluating time: naive implementation
#             flip_energies = []
#             for d in range(samples.shape[-1]):
#                 flip_samples_d = samples.clone()
#                 flip_samples_d[:,d] = 1 - flip_samples_d[:,d]
#                 flip_energies.append(model(flip_samples_d))
                
#             energy_xp = torch.cat(flip_energies, dim=1) ### [128*32, 32]
            


### save record for the first epoch
            if epoch == 0 and args.record_name is not None:
                energy_rand = model(rand_samples)
                train_records['energies'].append([torch.mean(energy_x).item(), torch.std(energy_x).item(), 
                                                  torch.mean(energy_xp).item(), torch.std(energy_xp).item(),
                                                  torch.mean(energy_rand).item(), torch.std(energy_rand).item()])

            loss = torch.norm(torch.exp(energy_x-energy_xp), dim=-1)**2
    
            
       

            loss = loss.mean()
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            
            pbar.set_description('Epoch: %d, Loss: %.2f' % (epoch, loss.item()))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        print("Duration:", t_end - t_start)
        print('Energy (positive samples): %.2f±%.2f, Energy (random samples): %.2f±%.2f' % (torch.mean(energy_x).item(), torch.std(energy_x).item(), torch.mean(model(rand_samples)).item(), torch.std(model(rand_samples)).item()))
        
        
        ### Compute mmd 
        if args.mmd:
            file =os.path.join(args.save_dir, 'mmd.txt')
            true_samples = float2bin_new(db.gen_batch(4000), args.discrete_dim, db.int_scale)
            true_samples = torch.from_numpy(true_samples).to(device).float()
            rand_samples = torch.randint(2, (4000, args.discrete_dim)).to(device).float()
            gibbs_sampler = GibbsSampler(2, args.discrete_dim, device)
            mmd, _ = estimate_hamming_new(model, true_samples, rand_samples, gibbs_sampler, args.num_rounds_gibbs)
            print('MMD score: %.5f' % (mmd))
            print('==========================================')
            with open(file, 'a+') as f:
                f.write('Epoch: %d, MMD: %.5f' % (epoch, mmd))
                f.write('\n')
         
        
        ### Save record for positive sample energy 
        if args.record_name is not None:
            energy_xp = torch.reshape(energy_xp, (-1, )) 
            energy_rand = model(rand_samples)
            train_records['energies'].append([torch.mean(energy_x).item(), torch.std(energy_x).item(), 
                                              torch.mean(energy_xp).item(), torch.std(energy_xp).item(),
                                              torch.mean(energy_rand).item(), torch.std(energy_rand).item()])
            train_records['loss'].append(sum(losses) / len(losses))

            with open(args.save_dir+'/{}.json'.format(args.record_name), 'w') as fp:
                json.dump(train_records, fp)

        ### Save checkpoints
        if (epoch+1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, '{}-epoch_{}.pt'.format(args.data_name, epoch+1)))
            print('Saving checkpoint at epoch ', epoch+1)
            if args.plot:
                plot_heat(model, device, args.discrete_dim, db.int_scale, out_file=os.path.join(args.save_dir, 'heat-{}-epoch_{}.png'.format(args.data_name, epoch+1)))
                print('Saving heat map at epoch ', epoch+1)
            print('==========================================')
    
if __name__ == '__main__':
    tab_printer(args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
    
    ### Set up dataset
    print('Set up data...')
    db = OnlineToyDataset(args.data_name, args.discrete_dim)
    print('Data is ready!')
    
    
    ### Initialize model
    model = EBM_MLP(args.discrete_dim, args.hidden_dim, bn=args.bn, activation=args.activation) #, std=args.std)
    print('==========================================')
    print(model)
    model = model.to(device)
    
    if not os.path.exists(os.path.join(args.save_dir)):
        os.makedirs(os.path.join(args.save_dir))
        
    ### Visulize true samples
    if args.plot_data:
        samples = db.gen_batch(10000)
        plot_samples(samples, out_file=os.path.join(args.save_dir, '{}-data.png'.format(args.data_name)))
        print('Visualization of data samples.')
        print('==========================================')
        sys.exit()
        
    ### Train
    train(model, db, device)
