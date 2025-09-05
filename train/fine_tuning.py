from models.mlp import MLP
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from utils.training import train, test

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, help="Model file name")
  parser.add_argument("--database", type=str, choices={"mnist", "cifar10"}, default="mnist", help="Mnist or Cifar10")
  parser.add_argument('--batch_size', type=int, default=512)
  parser.add_argument('--epochs', type=int, default = 3)
  parser.add_argument("--lr", type=float, default = 1e-4)
  parser.add_argument('--log-interval', type=int, default=50, help='how many batches to wait before logging training status')
  args = parser.parse_args()

  # Get data
  args = parser.parse_args()
  use_cuda = torch.cuda.is_available()

  device = torch.device("cuda" if use_cuda else "cpu")

  train_kwargs = {'batch_size': args.batch_size}
  test_kwargs = {'batch_size': args.batch_size}
  if use_cuda:
      cuda_kwargs = {'num_workers': 2,
                      'pin_memory': True,
                      'shuffle': True}
      train_kwargs.update(cuda_kwargs)
      test_kwargs.update(cuda_kwargs)
  if args.database == "mnist":
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                        transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

  state_dict = torch.load(args.model, map_location=device)
  model = MLP().to(device)
  model.load_state_dict(state_dict)

  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  for epoch in range(1, args.epochs + 1):
      train(args, model, device, train_loader, optimizer, epoch)
      test(model, device, test_loader)

  torch.save(model.state_dict(), f"{args.database}_mlp_finetuned.pt")


if __name__ == "__main__":
  main()