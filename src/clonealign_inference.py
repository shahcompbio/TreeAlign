import torch
import pandas as pd
import numpy as np
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
def clonealign_pyro_gene_model(cnv, expr, temperature=0.5):
    num_of_clones = len(cnv)
    num_of_cells = len(expr)
    num_of_genes = len(expr[0])

    softplus = Softplus()

    # initialize per_copy_expr using the data (This typically speeds up convergence)
    expr = expr * 3000 / torch.reshape(torch.sum(expr, 1), (num_of_cells, 1))
    per_copy_expr_guess = torch.mean(expr, 0)

    # draw chi from gamma
    chi = pyro.sample('expose_chi', dist.Gamma(torch.ones(6) * 2, torch.ones(6)).to_event(1))

    with pyro.plate('gene', num_of_genes):
        # draw per_copy_expr from softplus-transformed Normal distribution
        per_copy_expr = pyro.sample('expose_per_copy_expr',
                                    dist.Normal(inverse_softplus(per_copy_expr_guess), torch.ones(num_of_genes)))
        per_copy_expr = softplus(per_copy_expr)

        # draw mean_expr from another softplus-transformed Normal distribution
        mean_expr = pyro.sample('expose_mean_expr',
                                dist.Normal(inverse_softplus(per_copy_expr_guess), torch.ones(num_of_genes)))
        mean_expr = softplus(mean_expr)

        # draw w from Normal
        w = pyro.sample('expose_w', dist.Normal(torch.zeros(6), torch.sqrt(chi)).to_event(1))

        # sample the gene_type_score from uniform distribution.
        # the score reflects how much the copy number influence expression.
        gene_type_score = pyro.sample('expose_gene_type_score', dist.Dirichlet(torch.ones(2)))
        gene_type = pyro.sample('expose_gene_type',
                                dist.RelaxedOneHotCategorical(temperature=torch.tensor(temperature), probs=gene_type_score))

    with pyro.plate('cell', num_of_cells):
        # draw clone_assign_prob from Dir
        clone_assign_prob = pyro.sample('expose_clone_assign_prob', dist.Dirichlet(torch.ones(num_of_clones)))
        # draw clone_assign from Cat
        clone_assign = pyro.sample('clone_assign', dist.Categorical(clone_assign_prob))

        # draw psi from Normal
        psi = pyro.sample('expose_psi', dist.Normal(torch.zeros(6), torch.ones(6)).to_event(1))

        # construct expected_expr
        expected_expr = (per_copy_expr * Vindex(cnv)[clone_assign] * gene_type[:, 0] +
                         mean_expr * gene_type[:, 1]) * \
                        torch.exp(torch.matmul(psi, torch.transpose(w, 0, 1)))

        # draw expr from Multinomial
        pyro.sample('obs', dist.Multinomial(total_count=3000, probs=expected_expr, validate_args=False), obs=expr)


@config_enumerate
def clonealign_pyro_fc_model(cnv, expr, temperature=0.5):
    num_of_clones = len(cnv)
    num_of_cells = len(expr)
    num_of_genes = len(expr[0])

    softplus = Softplus()

    # initialize per_copy_expr using the data (This typically speeds up convergence)
    expr = expr * 3000 / torch.reshape(torch.sum(expr, 1), (num_of_cells, 1))
    per_copy_expr_guess = torch.mean(expr, 0)

    cnv_min = torch.min(cnv, 0)[0]
    cnv_rel = cnv / cnv_min

    # draw chi from gamma
    chi = pyro.sample('expose_chi', dist.Gamma(torch.ones(6) * 2, torch.ones(6)).to_event(1))

    with pyro.plate('gene', num_of_genes):
        # draw per_copy_expr from softplus-transformed Normal distribution
        per_copy_expr = pyro.sample('expose_per_copy_expr',
                                    dist.Normal(inverse_softplus(per_copy_expr_guess), torch.ones(num_of_genes)))
        per_copy_expr = softplus(per_copy_expr)

        # draw w from Normal
        w = pyro.sample('expose_w', dist.Normal(torch.zeros(6), torch.sqrt(chi)).to_event(1))
        # draw gene fold change
        gene_fold_change = pyro.sample('expose_gene_fold_change', dist.Uniform(low = 0, high = 10))


    with pyro.plate('cell', num_of_cells):
        # draw clone_assign_prob from Dir
        clone_assign_prob = pyro.sample('expose_clone_assign_prob', dist.Dirichlet(torch.ones(num_of_clones)))
        # draw clone_assign from Cat
        clone_assign = pyro.sample('clone_assign', dist.Categorical(clone_assign_prob))

        # draw psi from Normal
        psi = pyro.sample('expose_psi', dist.Normal(torch.zeros(6), torch.ones(6)).to_event(1))

        # construct expected_expr
        expected_expr = (per_copy_expr * torch.pow(Vindex(cnv_rel)[clone_assign], gene_fold_change) * cnv_min) * torch.exp(
            torch.matmul(psi, torch.transpose(w, 0, 1)))

        # draw expr from Multinomial
        pyro.sample('obs', dist.Multinomial(total_count=3000, probs=expected_expr, validate_args=False), obs=expr)


@config_enumerate
def clonealign_pyro_model(cnv, expr, temperature=0.5):
    num_of_clones = len(cnv)
    num_of_cells = len(expr)
    num_of_genes = len(expr[0])

    softplus = Softplus()

    # initialize per_copy_expr using the data (This typically speeds up convergence)
    expr = expr * 3000 / torch.reshape(torch.sum(expr, 1), (num_of_cells, 1))
    per_copy_expr_guess = torch.mean(expr, 0)

    # calculate copy number mean
    copy_number_mean = torch.mean(cnv, 0)

    # draw chi from gamma
    chi = pyro.sample('expose_chi', dist.Gamma(torch.ones(6) * 2, torch.ones(6)).to_event(1))

    with pyro.plate('gene', num_of_genes):
        # draw per_copy_expr from softplus-transformed Normal distribution
        per_copy_expr = pyro.sample('expose_per_copy_expr',
                                    dist.Normal(inverse_softplus(per_copy_expr_guess), torch.ones(num_of_genes)))
        per_copy_expr = softplus(per_copy_expr)

        # draw w from Normal
        w = pyro.sample('expose_w', dist.Normal(torch.zeros(6), torch.sqrt(chi)).to_event(1))

    with pyro.plate('cell', num_of_cells):
        # draw clone_assign_prob from Dir
        clone_assign_prob = pyro.sample('expose_clone_assign_prob', dist.Dirichlet(torch.ones(num_of_clones)))
        # draw clone_assign from Cat
        clone_assign = pyro.sample('clone_assign', dist.Categorical(clone_assign_prob))

        # draw psi from Normal
        psi = pyro.sample('expose_psi', dist.Normal(torch.zeros(6), torch.ones(6)).to_event(1))

        # construct expected_expr
        expected_expr = per_copy_expr * Vindex(cnv)[clone_assign] * torch.exp(
            torch.matmul(psi, torch.transpose(w, 0, 1)))

        # draw expr from Multinomial
        pyro.sample('obs', dist.Multinomial(total_count=3000, probs=expected_expr, validate_args=False), obs=expr)


def run_clonealign_pyro(cnv, expr, model_select=""):
    tau0 = 1.0
    ANNEAL_RATE = 0.001
    MIN_TEMP = 0.5
    np_temp = tau0
    losses = []
    max_iter = 500
    rel_tol = 1e-5

    optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
    elbo = TraceEnum_ELBO(max_plate_nesting=1)

    if model_select == "gene":
        model = clonealign_pyro_gene_model
    elif model_select == "fc":
        model = clonealign_pyro_fc_model
    else:
        model = clonealign_pyro_model

    pyro.clear_param_store()

    global_guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("expose_")))


    svi = SVI(model, global_guide, optim, loss=elbo)


    # start inference
    print('Start Inference.')
    for i in range(max_iter):
        if i % 100 == 1:
            np_temp = np.maximum(tau0 * np.exp(-ANNEAL_RATE * i), MIN_TEMP)
        loss = svi.step(cnv, expr, np_temp)

        if i >= 1:
            loss_diff = abs((losses[-1] - loss) / losses[-1])
            if loss_diff < rel_tol:
                print('ELBO converged at iteration ' + str(i))
                break

        losses.append(loss)
        print('.' if i % 200 else '\n', end='')

    map_estimates = global_guide(cnv, expr)

    clone_assign_prob = map_estimates['expose_clone_assign_prob']
    gene_type_score_df = None
    gene_fold_change_df = None

    if model_select == "gene":
        gene_type_score = map_estimates['expose_gene_type_score']
        gene_type_score_df = pd.DataFrame(gene_type_score.data.numpy())
    elif model_select == "fc":
        gene_fold_change = map_estimates['expose_gene_fold_change']
        gene_fold_change_df = pd.DataFrame(gene_fold_change.data.numpy())

    clone_assign_prob_df = pd.DataFrame(clone_assign_prob.data.numpy())

    return clone_assign_prob_df, gene_type_score_df, gene_fold_change_df
