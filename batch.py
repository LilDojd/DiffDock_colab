import os

import copy
import os
import torch

import time
from argparse import ArgumentParser, Namespace, FileType
from rdkit.Chem import RemoveHs
from functools import partial
import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, AddHs
from torch_geometric.loader import DataLoader
import yaml
import sys
import csv

csv.field_size_limit(sys.maxsize)

print(torch.__version__)
os.makedirs("data/esm2_output", exist_ok=True)
os.makedirs("results", exist_ok=True)
from datasets.process_mols import (
    read_molecule,
    generate_conformer,
    write_mol_with_coords,
)
from datasets.pdbbind import PDBBind
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.visualise import PDBFile
from tqdm import tqdm
from esm_embedding_preparation import esm_embedding_prep
import subprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(f"/apps/DiffDock/workdir/paper_score_model/model_parameters.yml") as f:
    score_model_args = Namespace(**yaml.full_load(f))

with open(f"/apps/DiffDock/workdir/paper_confidence_model/model_parameters.yml") as f:
    confidence_args = Namespace(**yaml.full_load(f))

import shutil

t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
state_dict = torch.load(
    f"/apps/DiffDock/workdir/paper_score_model/best_ema_inference_epoch_model.pt",
    map_location=torch.device("cpu"),
)
model.load_state_dict(state_dict, strict=True)
model = model.to(device)
model.eval()

confidence_model = get_model(
    confidence_args,
    device,
    t_to_sigma=t_to_sigma,
    no_parallel=True,
    confidence_mode=True,
)
state_dict = torch.load(
    f"/apps/DiffDock/workdir/paper_confidence_model/best_model_epoch75.pt",
    map_location=torch.device("cpu"),
)
confidence_model.load_state_dict(state_dict, strict=True)
confidence_model = confidence_model.to(device)
confidence_model.eval()

import sys


def esm(protein_path, out_file):
    print("running esm")
    esm_embedding_prep(out_file, protein_path)
    # create args object with defaults
    os.environ["HOME"] = "/apps/DiffDock/esm/model_weights"
    subprocess.call(
        f"python -m extract esm2_t33_650M_UR50D {out_file} data/esm2_output --repr_layers 33 --include per_tok",
        shell=True,
        env=os.environ,
    )

if __name__=='__main__':
    df = pd.read_csv('protein_ligand.csv')

