# based on clonealign framework, re-write with pyro
import logging
import sys

import os
from collections import defaultdict
import torch
import numpy as np
import pandas as pd
import scipy.stats
from torch.distributions import constraints
from torch.nn import Softplus

import pyro
import pyro.distributions as dist
from pyro.distributions.util import broadcast_shape
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.ops.indexing import Vindex

smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.5.1')
pyro.enable_validation(True)

# input data

expr_input = "data/SPECTRUM-OV-022_expr_clonealign_input.csv"
cnv_input = "data/SPECTRUM-OV-022_cnv_clonealign_input.csv"

clone_assign_prob_output = "data/SPECTRUM-OV-022_clone_assign_prob_0.csv"
gene_type_score_output = "data/SPECTRUM-OV-022_gene_type_score_0.csv"


expr_csv = pd.read_csv(expr_input, header = 0, index_col=0)
cnv_csv = pd.read_csv(cnv_input, header = 0, index_col=0)

expr_csv = expr_csv[expr_csv.mean(1) > 0]
cnv_csv = cnv_csv.loc[expr_csv.index, ]

# cast cnv greater than 6
cnv = torch.tensor(cnv_csv.values, dtype=torch.float)
cnv = torch.transpose(cnv, 0, 1)

cnv[cnv > 6] = 6

expr = torch.tensor(expr_csv.values, dtype = torch.float)
expr = torch.transpose(expr, 0, 1)

expr = expr[0:10]
expr = expr[:, torch.mean(expr, dim = 0) > 0]

cnv = cnv[:, range(expr.shape[1])]

# input data: cnv, expr
# cnv: clone_count * gene_count
# expr: cell_count * gene_count

# use the 
def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))

@config_enumerate
def clonealign_pyro(cnv, expr):
    num_of_clones = len(cnv)
    num_of_genes = len(cnv[0])
    num_of_cells = len(expr)

    softplus = Softplus()

    # initialize per_copy_expr using the data (This typically speeds up convergence)
    expr = expr * 2000 / torch.reshape(torch.sum(expr, 1), (num_of_cells, 1))
    per_copy_expr_guess = torch.mean(expr, 0)

    # calculate copy number mean
    copy_number_mean = torch.mean(cnv, 0)

    # draw chi from gamma
    chi = pyro.sample('expose_chi', dist.Gamma(torch.ones(6) * 2, torch.ones(6)).to_event(1))

    gene_plate = pyro.plate('gene', num_of_genes, dim=-1)
    cell_plate = pyro.plate('cell', num_of_cells)

    with gene_plate:
        # draw per_copy_expr from softplus-transformed Normal distribution
        per_copy_expr = pyro.sample('expose_per_copy_expr',
                                    dist.Normal(inverse_softplus(per_copy_expr_guess), torch.ones(num_of_genes)))


        per_copy_expr = softplus(per_copy_expr)

        # instead of softplus-transformed normal, use negative binomial instead for per_copy_expr
        # per_copy_expr = pyro.sample('per_copy_expr', dist.NegativeBinomial())

        # draw w from Normal
        w = pyro.sample('expose_w', dist.Normal(torch.zeros(6), torch.sqrt(chi)).to_event(1))

        # sample the gene_type_score from uniform distribution.
        # the score reflects how much the copy number influence expression.
        gene_type_score = pyro.sample('expose_gene_type_score', dist.Beta(0.5, 0.5))
        gene_type = pyro.sample('gene_type', dist.Bernoulli(gene_type_score), infer={"enumerate": "parallel"})

    for i in cell_plate:
        # draw clone_assign_prob from Dir
        clone_assign_prob = pyro.sample('expose_clone_assign_prob_{}'.format(i), dist.Dirichlet(torch.ones(num_of_clones)))

        # draw clone_assign from Cat
        clone_assign = pyro.sample('clone_assign_{}'.format(i), dist.Categorical(clone_assign_prob), infer={"enumerate": "parallel"})

        # draw psi from Normal
        psi = pyro.sample('expose_psi_{}'.format(i), dist.Normal(torch.zeros(6), torch.ones(6)).to_event(1))

        with gene_plate:
            test1 = (Vindex(cnv)[torch.squeeze(clone_assign, -1)] * gene_type + copy_number_mean * (1 - gene_type))
            test2 = torch.exp(torch.matmul(torch.squeeze(psi), torch.transpose(w, 0, 1)))
            test3 = per_copy_expr

            expected_expr = test1 * test2 * test3
            fake_probs = expected_expr.new_zeros(broadcast_shape(expected_expr.shape)) + 0.5              

            # draw expr from negative binomial
            pyro.sample('obs_{}'.format(i), dist.NegativeBinomial(total_count = expected_expr * 1000, probs=fake_probs, validate_args=False), obs=expr[i])

def guide(cnv, expr):
    num_of_clones = len(cnv)
    num_of_genes = len(cnv[0])
    num_of_cells = len(expr)

    chi_loc = pyro.param('chi_loc', 
                        lambda: torch.ones(6) * 0.1,
                        constraint=constraints.positive)
    chi = pyro.sample('expose_chi', dist.Delta(chi_loc).to_event(1))


    gene_plate = pyro.plate('gene', num_of_genes, dim=-1)
    cell_plate = pyro.plate('cell', num_of_cells, subsample_size = 5)

    per_copy_expr_loc = pyro.param('per_copy_expr_loc', 
                                  lambda: torch.exp(torch.randn(num_of_genes)))

    w_loc = pyro.param('w_loc', 
                       lambda: torch.randn(num_of_genes, 6))

    gene_type_score_loc = pyro.param('gene_type_score_loc',
                                lambda: torch.rand(num_of_genes))

    clone_assign_prob_loc = pyro.param('clone_assign_prob_loc',
                                  lambda: torch.rand(num_of_cells, num_of_clones),
                                      constraint = constraints.simplex)

    psi_loc = pyro.param('psi_loc',
                                 lambda: torch.randn(num_of_cells, 6))    

    with gene_plate:
        # draw per_copy_expr from softplus-transformed Normal distribution
        per_copy_expr = pyro.sample('expose_per_copy_expr', dist.Delta(per_copy_expr_loc))
        w = pyro.sample('expose_w', dist.Delta(w_loc).to_event(1))
        gene_type_score = pyro.sample('expose_gene_type_score', dist.Delta(gene_type_score_loc))


    for i in cell_plate:
        # draw clone_assign_prob from Dir
        clone_assign_prob = pyro.sample('expose_clone_assign_prob_{}'.format(i), dist.Delta(clone_assign_prob_loc[i]).to_event(1))
        # draw psi from Normal
        psi = pyro.sample('expose_psi_{}'.format(i), dist.Delta(psi_loc[i]).to_event(1))     

# initialize Adam optimizer
optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})

# TraceEnum_ELBO will marginalize out the assignments of datapoints to clusters
elbo = TraceEnum_ELBO(max_plate_nesting=3)

pyro.clear_param_store()

# AutoGuide
global_guide = guide
# put together SVI object
svi = SVI(clonealign_pyro, global_guide, optim, loss=elbo)

import datetime

print(datetime.datetime.now())

gradient_norms = defaultdict(list)
print(svi.loss(clonealign_pyro, global_guide, cnv, expr))  # Initializes param store.

print(datetime.datetime.now())

