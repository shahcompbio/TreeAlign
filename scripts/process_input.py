# based on clonealign framework, re-write with pyro
import torch
import pandas as pd


def process_input(expr_path, cnv_path):
    expr_csv = pd.read_csv(expr_path, header = 0, index_col=0)
    cnv_csv = pd.read_csv(cnv_path, header = 0, index_col=0)

    # normalized the by minCNV
    cnv_csv.div(cnv_csv.min(axis=1), axis=0)

    # cast cnv greater than 6
    cnv = torch.tensor(cnv_csv.values, dtype=torch.float)
    cnv = torch.transpose(cnv, 0, 1)

    cnv[cnv > 6] = 6

    expr = torch.tensor(expr_csv.values, dtype = torch.float)
    expr = torch.transpose(expr, 0, 1)

    return expr, cnv, expr_csv, cnv_csv