import torch
import pandas as pd


def process_input_matrices(expr_path, cnv_path, cnv_cutoff=6):
    expr_csv = pd.read_csv(expr_path, header=0, index_col=0)
    cnv_csv = pd.read_csv(cnv_path, header=0, index_col=0)

    # remove genes that are not in the cnv region
    cnv_csv = cnv_csv[cnv_csv.var(1) > 0]
    expr_csv = expr_csv.loc[cnv_csv.index,]

    cnv = torch.tensor(cnv_csv.values, dtype=torch.float)
    cnv = torch.transpose(cnv, 0, 1)

    cnv[cnv > cnv_cutoff] = cnv_cutoff

    expr = torch.tensor(expr_csv.values, dtype=torch.float)
    expr = torch.transpose(expr, 0, 1)

    return expr, cnv, expr_csv, cnv_csv


def process_output_matrices(clone_prob, gene_type_score, clone_prob_path, gene_type_score_path, expr, cnv):
    # set the rownames and colnames of clone_prob
    clone_prob_rownames = {i: c for i, c in enumerate(expr.columns)}
    clone_prob_colnames = {i: c for i, c in enumerate(cnv.columns)}
    clone_prob.rename(index=clone_prob_rownames, inplace=True)
    clone_prob.rename(columns=clone_prob_colnames, inplace=True)

    gene_type_score_rownames = {i:c for i, c in enumerate(expr.index)}
    if gene_type_score != None:
        gene_type_score.rename(index=gene_type_score_rownames, inplace=True)
        pd.DataFrame(gene_type_score.to_csv(gene_type_score_path))

    pd.DataFrame(clone_prob.to_csv(clone_prob_path))

