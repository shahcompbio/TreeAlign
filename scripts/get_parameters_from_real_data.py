import torch

from torch.nn import Softplus
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.ops.indexing import Vindex
import random


# use the
def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


@config_enumerate
def clonealign_pyro(cnv, expr):
    num_of_clones = len(cnv)
    num_of_cells = len(expr)
    num_of_genes = len(expr[0])

    softplus = Softplus()

    # initialize per_copy_expr using the data (This typically speeds up convergence)
    expr = expr * 2000 / torch.reshape(torch.sum(expr, 1), (num_of_cells, 1))
    per_copy_expr_guess = torch.mean(expr, 0)

    # draw chi from gamma
    chi = pyro.sample('chi', dist.Gamma(torch.ones(6) * 2, torch.ones(6)).to_event(1))

    with pyro.plate('gene', num_of_genes):
        # draw per_copy_expr from softplus-transformed Normal distribution
        per_copy_expr = pyro.sample('per_copy_expr',
                                    dist.Normal(inverse_softplus(per_copy_expr_guess), torch.ones(num_of_genes)))
        per_copy_expr = softplus(per_copy_expr)

        # draw mean_expr from another softplus-transformed Normal distribution
        mean_expr = pyro.sample('mean_expr',
                                dist.Normal(inverse_softplus(per_copy_expr_guess), torch.ones(num_of_genes)))
        mean_expr = softplus(mean_expr)

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
        expected_expr = (per_copy_expr * Vindex(cnv)[clone_assign] * gene_type_score[:, 0] +
                         mean_expr * gene_type_score[:, 1]) * \
                        torch.exp(torch.matmul(psi, torch.transpose(w, 0, 1)))

        # draw expr from Multinomial
        pyro.sample('obs', dist.Multinomial(total_count=2000, probs=expected_expr, validate_args=False), obs=expr)


def get_parameters(expr, cnv):
    # initialize Adam optimizer
    optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})

    # TraceEnum_ELBO will marginalize out the assignments of datapoints to clusters
    elbo = TraceEnum_ELBO(max_plate_nesting=1)

    pyro.clear_param_store()

    # AutoGuide
    global_guide = AutoDelta(poutine.block(clonealign_pyro,
                                           expose=['chi', 'per_copy_expr', 'mean_expr', 'w', 'k', 'gene_type_score',
                                                   'clone_assign_prob', 'psi']))
    # put together SVI object
    svi = SVI(clonealign_pyro, global_guide, optim, loss=elbo)

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

    per_copy_expr = map_estimates['per_copy_expr']
    mean_expr = map_estimates['mean_expr']
    psi = map_estimates['psi']
    w = map_estimates['w']

    return cnv, expr, per_copy_expr, mean_expr, psi, w


def clonealign_pyro_simulation(cnv, expr, per_copy_expr, psi, w, gene_type_freq, cell_count, gene_count):
    # randomly select a given number of cells and genes for expr matrix
    random_cells = random.sample(range(len(expr)), cell_count)
    random_genes = random.sample(range(len(expr[0])), gene_count)


    cnv = cnv[:, random_genes]
    expr = expr[random_cells, :]
    expr = expr[:, random_genes]
    per_copy_expr = per_copy_expr[random_genes]

    w = w[random_genes, :]
    psi = psi[random_cells, :]


    num_of_clones = len(cnv)
    num_of_cells = len(expr)
    num_of_genes = len(expr[0])

    softplus = Softplus()

    # calculate copy number mean
    per_copy_expr = softplus(per_copy_expr)

    # simulate gene_type_scores
    gene_type_score_0 = torch.zeros(num_of_genes)
    gene_type_score_1 = torch.ones(num_of_genes)

    gene_type_score_0 = torch.reshape(gene_type_score_0, (num_of_genes, 1))
    gene_type_score_1 = torch.reshape(gene_type_score_1, (num_of_genes, 1))

    gene_type_score = torch.cat((gene_type_score_0, gene_type_score_1), 1)

    gene_type_count = int(num_of_genes * gene_type_freq * 0.01)
    random_indices = random.sample(range(num_of_genes), gene_type_count)
    gene_type_score[random_indices, 0] = 1
    gene_type_score[random_indices, 1] = 0

    with pyro.plate('cell', num_of_cells):
        # draw clone_assign_prob from Dir
        clone_assign_prob = pyro.sample('clone_assign_prob', dist.Dirichlet(torch.ones(num_of_clones)))
        # draw clone_assign from Cat
        clone_assign = pyro.sample('clone_assign', dist.Categorical(clone_assign_prob))

        expected_expr = (per_copy_expr * Vindex(cnv)[clone_assign] * gene_type_score[:, 0] +
                         per_copy_expr * gene_type_score[:, 1]) * \
                        torch.exp(torch.matmul(psi, torch.transpose(w, 0, 1)))

        # draw expr from Multinomial
        expr_simulated = pyro.sample('obs',
                                     dist.Multinomial(total_count=2000, probs=expected_expr, validate_args=False))

    return expr_simulated, gene_type_score, clone_assign, random_cells, random_genes
