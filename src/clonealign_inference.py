import torch
import pandas as pd
from torch.nn import Softplus

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.ops.indexing import Vindex


def inverse_softplus(x):
    '''
    inverse the softplus function
    :param x: number matrix
    :return:
    '''
    return x + torch.log(-torch.expm1(-x))


@config_enumerate
def clonealign_pyro_gene_model(cnv, expr):
    num_of_clones = len(cnv)
    num_of_cells = len(expr)
    num_of_genes = len(expr[0])

    softplus = Softplus()

    # initialize per_copy_expr using the data (This typically speeds up convergence)
    per_copy_expr_guess = torch.mean(expr / torch.reshape(torch.sum(expr, 1), (num_of_cells, 1)), 0)

    # calculate copy number mean
    copy_number_mean = torch.mean(cnv, 0)

    # draw chi from gamma
    chi = pyro.sample('chi', dist.Gamma(torch.ones(6) * 2, torch.ones(6)).to_event(1))

    with pyro.plate('gene', num_of_genes):
        # draw per_copy_expr from softplus-transformed Normal distribution
        per_copy_expr = pyro.sample('per_copy_expr',
                                    dist.Normal(inverse_softplus(per_copy_expr_guess), torch.ones(num_of_genes)))
        per_copy_expr = softplus(per_copy_expr)

        # draw w from Normal
        w = pyro.sample('w', dist.Normal(torch.zeros(6), torch.sqrt(chi)).to_event(1))

        # sample the gene_type_score from uniform distribution.
        # the score reflects how much the copy number influence expression.
        gene_type_score = pyro.sample('gene_type_score', dist.Dirichlet(torch.ones(2) * 0.1))


    with pyro.plate('cell', num_of_cells):
        # draw clone_assign_prob from Dir
        clone_assign_prob = pyro.sample('clone_assign_prob', dist.Dirichlet(torch.ones(num_of_clones)))
        # draw clone_assign from Cat
        clone_assign = pyro.sample('clone_assign', dist.Categorical(clone_assign_prob))

        # draw psi from Normal
        psi = pyro.sample('psi', dist.Normal(torch.zeros(6), torch.ones(6)).to_event(1))

        # construct expected_expr
        expected_expr = per_copy_expr * (
                    Vindex(cnv)[clone_assign] * gene_type_score[:, 0] + copy_number_mean * gene_type_score[:, 1]) * torch.exp(
            torch.matmul(psi, torch.transpose(w, 0, 1)))

        # draw expr from Multinomial
        pyro.sample('obs', dist.Multinomial(probs=expected_expr, validate_args=False), obs=expr)


@config_enumerate
def clonealign_pyro_model(cnv, expr):
    num_of_clones = len(cnv)
    num_of_cells = len(expr)
    num_of_genes = len(expr[0])

    softplus = Softplus()

    # initialize per_copy_expr using the data (This typically speeds up convergence)
    per_copy_expr_guess = torch.mean(expr / torch.reshape(torch.sum(expr, 1), (num_of_cells, 1)), 0)

    # calculate copy number mean
    copy_number_mean = torch.mean(cnv, 0)

    # draw chi from gamma
    chi = pyro.sample('chi', dist.Gamma(torch.ones(6) * 2, torch.ones(6)).to_event(1))

    with pyro.plate('gene', num_of_genes):
        # draw per_copy_expr from softplus-transformed Normal distribution
        per_copy_expr = pyro.sample('per_copy_expr',
                                    dist.Normal(inverse_softplus(per_copy_expr_guess), torch.ones(num_of_genes)))
        per_copy_expr = softplus(per_copy_expr)

        # draw w from Normal
        w = pyro.sample('w', dist.Normal(torch.zeros(6), torch.sqrt(chi)).to_event(1))


    with pyro.plate('cell', num_of_cells):
        # draw clone_assign_prob from Dir
        clone_assign_prob = pyro.sample('clone_assign_prob', dist.Dirichlet(torch.ones(num_of_clones)))
        # draw clone_assign from Cat
        clone_assign = pyro.sample('clone_assign', dist.Categorical(clone_assign_prob))

        # draw psi from Normal
        psi = pyro.sample('psi', dist.Normal(torch.zeros(6), torch.ones(6)).to_event(1))

        # construct expected_expr
        expected_expr = per_copy_expr * Vindex(cnv)[clone_assign] * torch.exp(
            torch.matmul(psi, torch.transpose(w, 0, 1)))

        # draw expr from Multinomial
        pyro.sample('obs', dist.Multinomial(probs=expected_expr, validate_args=False), obs=expr)


def run_clonealign_pyro(cnv, expr, is_gene_type=False):
    optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
    elbo = TraceEnum_ELBO(max_plate_nesting=1)

    pyro.set_rng_seed(10)
    pyro.clear_param_store()

    if is_gene_type:
        global_guide = AutoDelta(poutine.block(clonealign_pyro_gene_model,
                                           expose=['gene_type_score', 'chi', 'per_copy_expr', 'w', 'clone_assign_prob',
                                                   'psi']))
        svi = SVI(clonealign_pyro_gene_model, global_guide, optim, loss=elbo)
    else:
        global_guide = AutoDelta(poutine.block(clonealign_pyro_model,
                                           expose=['chi', 'per_copy_expr', 'w', 'clone_assign_prob',
                                                   'psi']))
        svi = SVI(clonealign_pyro_model, global_guide, optim, loss=elbo)

    # start inference
    losses = []
    max_iter = 200
    rel_tol = 1e-5
    print('Start Inference.')
    for i in range(max_iter):
        loss = svi.step(cnv, expr)

        if i >= 1:
            loss_diff = abs((losses[-1] - loss) / losses[-1])
            if loss_diff < rel_tol:
                print('ELBO converged at iteration ' + str(i))
                break

        losses.append(loss)

        print('.' if i % 200 else '\n', end='')

    map_estimates = global_guide(cnv, expr)

    clone_assign_prob = map_estimates['clone_assign_prob']

    if is_gene_type:
        gene_type_score = map_estimates['gene_type_score']

    clone_assign_prob_df = pd.DataFrame(clone_assign_prob.data.numpy())
    if is_gene_type:
        gene_type_score_df = pd.DataFrame(gene_type_score.data.numpy())
    else:
        gene_type_score_df = None
    return clone_assign_prob_df, gene_type_score_df
