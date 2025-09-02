# build_activation_state_dict_two_files.py
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from models.mlp import MLP


def collect_activations_state_dict(model: nn.Module,
                                   loader: torch.utils.data.DataLoader,
                                   device: torch.device,
                                   post_activation: bool = False) -> "OrderedDict[str, torch.Tensor]":
    """
    Raccoglie le attivazioni (N, D) per TUTTI i layer Linear del modello e
    le restituisce in un dict con chiavi tipo 'layer0.weight', 'layer1.weight', ...
    """
    model.eval()
    acts_lists: "OrderedDict[str, list[torch.Tensor]]" = OrderedDict()

    def make_hook(name):
        def hook_fn(_, __, output):
            out = output
            out = F.relu(out)
            acts_lists.setdefault(f"{name}.weight", []).append(out.detach().cpu())
        return hook_fn

    # Registra hook su tutti i layer Linear (mantiene l'ordine di named_modules)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_hook(name)))

    # Forward su tutto il loader
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            _ = model(x)

    # Rimuovi hook
    for h in handles:
        h.remove()

    # Concatena le liste in tensori (N, D)
    acts_state_dict = OrderedDict()
    for k, chunks in acts_lists.items():
        acts_state_dict[k] = torch.cat(chunks, dim=0)  # (N, D)

    return acts_state_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="file state_dict modello A")
    ap.add_argument("--model-b", required=True, help="file state_dict modello B")
    ap.add_argument("--dataset", choices=["train", "test"], default="train", required=False)
    ap.add_argument("--n-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset MNIST stessa transform del training
    transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])
    
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    ds = train_ds if args.dataset == "train" else test_ds

    # Seleziona N campioni casuali (riproducibili)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[:args.n_samples]
    subset = torch.utils.data.Subset(ds, idx)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=500, shuffle=False, pin_memory=True)

    # Modelli
    model_a = MLP().to(device)
    model_b = MLP().to(device)

    checkpoint = torch.load(args.model_a, map_location=device)
    model_a.load_state_dict(checkpoint)

    checkpoint_b = torch.load(args.model_b, map_location=device)
    model_b.load_state_dict(checkpoint_b)

    # Raccogli attivazioni e salva due file separati
    acts_A = collect_activations_state_dict(model_a, loader, device, post_activation=args.post_activation)
    acts_B = collect_activations_state_dict(model_b, loader, device, post_activation=args.post_activation)

    torch.save(acts_A, "activations_A.pt")
    torch.save(acts_B, "activations_B.pt")


if __name__ == "__main__":
    main()
