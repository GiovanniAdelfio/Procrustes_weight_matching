# build_activation_state_dict_two_files.py
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from models.mlp import MLP
from utils.procrustes import Procrustes


def get_linear_layers(model: nn.Module):
    """Ritorna (names, modules) dei layer Linear in ordine di forward."""
    names, modules = [], []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            names.append(name)
            modules.append(m)
    return names, modules


@torch.no_grad()
def get_layer_activations(model: nn.Module,
                          layer: nn.Linear,
                          loader,
                          device,
                          apply_relu):
    """Estrae attivazioni (N, D_out) del layer specifico su tutti i batch del loader."""
    bufs = []

    def hook(_, __, out):
        out = F.relu(out) if apply_relu else out
        bufs.append(out.detach().cpu())

    handle = layer.register_forward_hook(hook)
    model.eval()
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
    handle.remove()

    A = torch.cat(bufs, dim=0).float()  # (N, D_out)

    #A = A - A.mean(axis=0, keepdims=True)
    return A  #  (N, D)


def safe_load(path, device):
    """Carica in modo sicuro uno state_dict (supporta PyTorch vecchi/nuovi)."""
    try:
        obj = torch.load(path, map_location=device, weights_only=True)  # PyTorch recente
    except TypeError:
        obj = torch.load(path, map_location=device)  # fallback
    return obj  # si assume sia direttamente lo state_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True, help="file state_dict modello A")
    ap.add_argument("--model_b", required=True, help="file state_dict modello B")
    ap.add_argument("--dataset", choices=["train", "test"], default="train", required=False)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--apply_relu", action="store_true")
    ap.add_argument("--model", default="mlp", choices=["mlp", "mlp_3"]) 
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
    loader = torch.utils.data.DataLoader(subset, batch_size=500,
                                         shuffle=False, pin_memory=True)

    # Modelli
    if args.model == "mlp":
        model_a = MLP().to(device)
        model_b = MLP().to(device)

    sd_a = safe_load(args.model_a, device)
    sd_b = safe_load(args.model_b, device)

    model_a.load_state_dict(sd_a)
    model_b.load_state_dict(sd_b)

    # Layer Linear in ordine
    names_a, layers_a = get_linear_layers(model_a)
    names_b, layers_b = get_linear_layers(model_b)
    assert [type(m) for m in layers_a] == [type(m) for m in layers_b], \
        "Strutture dei modelli A e B non corrispondono."

    # Indice ultimo layer da allineare
    last_align_idx = len(layers_b) - 2
    
        # --- Matching layer-per-layer con Procrustes ---
    for i in range(0, last_align_idx + 1):
        L_a = layers_a[i]
        L_b = layers_b[i]
        next_L_b = layers_b[i + 1] if i + 1 < len(layers_b) else None

        # 1) Attivazioni (stesso subset) — ricalcolate dopo ogni aggiornamento di B
        A = get_layer_activations(model_a, L_a, loader, device, args.apply_relu)
        B = get_layer_activations(model_b, L_b, loader, device, args.apply_relu)

        # 2) Procrustes: trova Q (D_out x D_out) tale che B Q ≈ A
        Q = Procrustes(B, A)  

        # 3) Applica Q al modello B (sinistra sul layer i, destra^T sul layer i+1)
        with torch.no_grad():
            # layer corrente: righe = uscite
            L_b.weight.copy_(Q.t() @ L_b.weight)
            if L_b.bias is not None:
                L_b.bias.copy_(Q.t() @ L_b.bias)

            # layer successivo: colonne = input (delle uscite del layer i)
            if next_L_b is not None:
                next_L_b.weight.copy_(next_L_b.weight @ Q)

    # Salva lo state_dict risultante del modello B
    torch.save(model_b.state_dict(), 'proc_activation_b.pt')
    print(f"Salvato modello B allineato in: proc_activation_b.pt")


if __name__ == "__main__":
    main()
