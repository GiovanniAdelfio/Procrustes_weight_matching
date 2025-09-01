from models.mlp import MLP
from utils.weight_matching import mlp_permutation_spec, weight_matching, apply_permutation
from utils.utils import flatten_params, lerp
from utils.plot import plot_interp_acc
import argparse
import torch
from torchvision import datasets, transforms
from utils.training import test
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from utils.weight_matching import Procustes, proc_weight_matching_MLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True)
    parser.add_argument("--model_b", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # load models
    model_a = MLP()
    model_b = MLP()
    checkpoint = torch.load(args.model_a)
    model_a.load_state_dict(checkpoint)   
    checkpoint_b = torch.load(args.model_b)
    model_b.load_state_dict(checkpoint_b)
    
    permutation_spec = mlp_permutation_spec(4)
    final_permutation = weight_matching(permutation_spec,
                                        flatten_params(model_a), flatten_params(model_b))
              

    updated_params = apply_permutation(permutation_spec, final_permutation, flatten_params(model_b))

    
    # test against mnist
    transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])
    test_kwargs = {'batch_size': 5000}
    train_kwargs = {'batch_size': 5000}
    dataset = datasets.MNIST('../data', train=False,
                      transform=transform)
    dataset1 = datasets.MNIST('../data', train=True, download=True,
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
    # naive
    model_b.load_state_dict(checkpoint_b)
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())
    for lam in tqdm(lambdas):
      naive_p = lerp(lam, model_a_dict, model_b_dict)
      model_b.load_state_dict(naive_p)
      test_loss, acc = test(model_b.cuda(), 'cuda', test_loader)
      test_acc_interp_naive.append(acc)
      train_loss, acc = test(model_b.cuda(), 'cuda', train_loader)
      train_acc_interp_naive.append(acc)

    # smart
    model_b.load_state_dict(updated_params)
    model_b.cuda()
    model_a.cuda()
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_b_dict = copy.deepcopy(model_b.state_dict())
    for lam in tqdm(lambdas):
      naive_p = lerp(lam, model_a_dict, model_b_dict)
      model_b.load_state_dict(naive_p)
      test_loss, acc = test(model_b.cuda(), 'cuda', test_loader)
      test_acc_interp_clever.append(acc)
      train_loss, acc = test(model_b.cuda(), 'cuda', train_loader)
      train_acc_interp_clever.append(acc)


    # procustes weight-matching 
    model_a.load_state_dict(checkpoint)
    model_b.load_state_dict(checkpoint_b)
    proc_dict = proc_weight_matching_MLP(checkpoint, checkpoint_b)
    
    model_a_dict = copy.deepcopy(model_a.state_dict())
    model_weights = [['layer0.weight', 'layer0.bias'], ['layer1.weight', 'layer1.bias'], ['layer2.weight', 'layer2.bias'], ['layer3.weight', 'layer3.bias'], ['layer4.weight', 'layer4.bias']]
    for num, layer in enumarate(['layer0.weight', 'layer1.weight', 'layer2.weight', 'layer3.weight']):
        Q = proc_dict[layer]
        for key in model_weights[num]:
            model_a_dict[key] = (model_a_dict[key].T @ Q).T
        if num <3 :
            for key in model_weights[num+1]:
                model_a_dict[key] = (Q.T @ model_a_dict[key].T).T

    model_b_dict = copy.deepcopy(model_b.state_dict())
    for lam in tqdm(lambdas):
      naive_p = lerp(lam, model_a_dict, model_b_dict)
      model_b.load_state_dict(naive_p)
      test_loss, acc = test(model_b.cuda(), 'cuda', test_loader)
      test_acc_interp_proc.append(acc)
      train_loss, acc = test(model_b.cuda(), 'cuda', train_loader)
      train_acc_interp_proc.append(acc)
            
    fig = plot_interp_acc(lambdas, train_acc_interp_naive, test_acc_interp_naive,
                    train_acc_interp_clever, test_acc_interp_clever, train_acc_interp_proc, test_acc_interp_proc)
    plt.savefig(f"mnist_mlp_weight_matching_interp_accuracy_epoch.png", dpi=300)

if __name__ == "__main__":
  main()
