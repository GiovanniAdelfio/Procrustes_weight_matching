from asyncio import subprocess
from models.mlp import MLP
from utils.weight_matching import mlp_permutation_spec, weight_matching, apply_permutation
from utils.utils import flatten_params, lerp, slerp_state_dict
from utils.plot import plot_interp_acc
import argparse
import torch
from torchvision import datasets, transforms
from utils.training import test
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from utils.procrustes import  proc_weight_matching_MLP, apply_alignment_MLP


'''
Compares interpolation accuracy between two models using three methods:
- PWM (Procrustes Weight Matching)
- SLERP (spherical interpolation)
- Git-Re-Basin permutation weight matching

Evaluates accuracy along the interpolation path for each method, plots the
results in a single figure, and saves it as a PNG file.
If post_activation: applies PAM (procrustes activation matching) after PWM before testing.
If fine_tune: performs two epochs of training, lr = 1e-4, with adam optimizer on model_b after PWM, before testing.
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True)
    parser.add_argument("--model_b", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--post_activation", action="store_true", help = "Applies activation matching after procrustes weight_matching") 
    parser.add_argument("--fine_tune", action="store_true", help = "Fine-tunes model B after procrustes weight_matching")
    args = parser.parse_args()

    # load models
    model_a = MLP()
    model_b = MLP()
    checkpoint = torch.load(args.model_a)
    model_a.load_state_dict(checkpoint)   
    checkpoint_b = torch.load(args.model_b)
    model_b.load_state_dict(checkpoint_b)

    # calculates and saves permutations for Git-Re-Basin permutation weight matching
    permutation_spec = mlp_permutation_spec(4)
    final_permutation = weight_matching(permutation_spec,
                                        flatten_params(model_a), flatten_params(model_b))
              

    updated_params = apply_permutation(permutation_spec, final_permutation, flatten_params(model_b))

    
    # loads and normalizes MNIST dataset, creates dataloaders
    transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])
    test_kwargs = {'batch_size': 5000}
    train_kwargs = {'batch_size': 5000}
    dataset = datasets.MNIST('../data', train=False, download = True,
                      transform=transform)
    dataset1 = datasets.MNIST('../data', train=True, download = True,
                      transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)                  
    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)
    lambdas = torch.linspace(0, 1, steps=25)

    test_acc_interp_clever = []
    test_acc_interp_naive = []
    test_acc_interp_proc = []
    train_acc_interp_clever = []
    train_acc_interp_naive = []
    train_acc_interp_proc = []


    # PWM with post activation and/or fine-tuning if required
    proc_dict = proc_weight_matching_MLP(checkpoint_b, checkpoint)
    model_b_dict = copy.deepcopy(model_b.cpu().state_dict())
    model_b_dict = apply_alignment_MLP(model_b_dict, proc_dict)


    torch.save(model_b_dict, 'procustes_aligned_B.pt')
    import subprocess, sys
    project_root = r"/content/Procrustes_weight_matching"             
    model = 'procustes_aligned_B'

    if args.post_activation:
      cmd = [
          sys.executable, "-m", "utils.activation_matching",
          "--model_a", args.model_a,
          "--model_b", model + ".pt",
          "--dataset", "train",
          "--n-samples", "5000",
          "--seed", "1"
      ]
      subprocess.run(cmd, check=True, cwd = project_root)
    
      model_b_dict = torch.load('proc_activation_B.pt', map_location='cpu')
      model = 'proc_activation_B'

    if args.fine_tune:
      cmd = [
          sys.executable, "-m", "train.fine_tuning",
          "--model", model + ".pt",
          "--database", "mnist",
          "--batch_size", "512",
          "--epochs", "2 ",
          "--lr", "1e-4",
          "--log-interval", "50"
      ]
      subprocess.run(cmd, check=True, cwd = project_root)
      model_b_dict = torch.load(model + '_mlp_finetuned.pt', map_location='cpu')


    model_a_dict = copy.deepcopy(model_a.cpu().state_dict())
    for lam in tqdm(lambdas):
      naive_p = lerp(lam, model_a_dict, model_b_dict)
      model_b.load_state_dict(naive_p)
      test_loss, acc = test(model_b.cuda(), 'cuda', test_loader)
      test_acc_interp_proc.append(acc)
      train_loss, acc = test(model_b.cuda(), 'cuda', train_loader)
      train_acc_interp_proc.append(acc)

    
    # SLERP 
    model_b.load_state_dict(checkpoint_b)
    model_a_dict = copy.deepcopy(model_a.cpu().state_dict())
    model_b_dict = copy.deepcopy(model_b.cpu().state_dict())
    for lam in tqdm(lambdas):
      #naive_p = lerp(lam, model_a_dict, model_b_dict)
      naive_p = slerp_state_dict(lam, model_a_dict, model_b_dict)
      model_b.load_state_dict(naive_p)
      test_loss, acc = test(model_b.cuda(), 'cuda', test_loader)
      test_acc_interp_naive.append(acc)
      train_loss, acc = test(model_b.cuda(), 'cuda', train_loader)
      train_acc_interp_naive.append(acc)

      

    # Git-Re-Basin permutation weight matching
    model_b.load_state_dict(updated_params)
    model_b.cuda()
    model_a.cuda()
    model_a_dict = copy.deepcopy(model_a.cpu().state_dict())
    model_b_dict = copy.deepcopy(model_b.cpu().state_dict())
    for lam in tqdm(lambdas):
      naive_p = lerp(lam, model_a_dict, model_b_dict)
      model_b.load_state_dict(naive_p)
      test_loss, acc = test(model_b.cuda(), 'cuda', test_loader)
      test_acc_interp_clever.append(acc)
      train_loss, acc = test(model_b.cuda(), 'cuda', train_loader)
      train_acc_interp_clever.append(acc)


    # saving the graph as png, name depends on wether we used fine-tuning and/or post-activation
    fig = plot_interp_acc(lambdas, train_acc_interp_naive, test_acc_interp_naive,
                    train_acc_interp_clever, test_acc_interp_clever, train_acc_interp_proc, test_acc_interp_proc)
    name ="mnist_mlp_weight_matching"
    if args.post_activation:
      name += "_post_activation"
    if args.fine_tune:
      name += "_finetuned"
    plt.savefig(name+".png", dpi=300)

if __name__ == "__main__":
  main()
