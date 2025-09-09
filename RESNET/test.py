
from pathlib import Path
from xml.parsers.expat import model


import jax.numpy as jnp
import wandb
from flax.serialization import from_bytes
from jax import random

from cifar10_resnet20_train import BLOCKS_PER_GROUP, ResNet, make_stuff
from utils import ec2_get_instance_type



with wandb.init(
      project="git-re-basin",  # sul tuo account
      tags=["cifar10", "resnet20", "weight-matching"],
      job_type="analysis",
  ) as wandb_run:
    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.model_a = "v12"
    config.load_epoch = 249
    filename = f"checkpoint{config.load_epoch}"

    model = ResNet(blocks_per_group=BLOCKS_PER_GROUP["resnet20"],
                   num_classes=10,
                   width_multiplier=1)
    stuff = make_stuff(model)


    def load_model(filepath):
        with open(filepath, "rb") as fh:
            return from_bytes(
                model.init(random.PRNGKey(0), jnp.zeros((1, 32, 32, 3)))["params"], fh.read())


    model_a = load_model(
        Path(wandb_run.use_artifact(
            f"skainswo/git-re-basin/cifar10-resnet-weights:{config.model_a}")
            .get_entry(filename).download()))
    

print(model_a.keys())
print(model_a['blockgroups_0'].keys())
print(model_a['blockgroups_0']['blocks_0'].keys())