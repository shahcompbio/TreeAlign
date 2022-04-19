"""
CloneAlign base class: initialize and clean expr, cnv data frame
"""

import torch
import numpy as np
import pandas as pd
from torch.nn import Softplus

import pyro
import pyro.optim
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.ops.indexing import Vindex


def max_count(s):
    '''
    calculate the number of appearances of the most common item in pandas.Series
    :param s: (pandas.Series)
    :return: (int)
    '''
    return s.value_counts(dropna=False).max()


def inverse_softplus(x):
    '''
    inverse the softplus function
    :param x: number matrix (torch.tensor)
    :return: number matrix (torch.tensor)
    '''
    return x + torch.log(-torch.expm1(-x))


class CloneAlign():

    def process_input_matrices(self):
        '''
        clean up clonealign input
        :return: None
        '''
        # remove genes that are not in the cnv region
        self.cnv_df = self.cnv_df[self.cnv_df.var(1) > 0]
        self.expr_df = self.expr_df[self.expr_df.mean(1) > 0]

        intersect_index = self.cnv_df.index.intersection(self.expr_df.index)

        self.expr_df = self.expr_df.loc[intersect_index,]
        self.cnv_df = self.cnv_df.loc[intersect_index,]

        self.cnv_df[self.cnv_df > self.cnv_cutoff] = self.cnv_cutoff

        return

    def generate_output(self):
        '''
        generate clone_assign_df and gene_type_score_df
        :return: clonealign output (pandas.DataFrame)
        '''
        clone_assign_df = pd.DataFrame.from_dict(self.clone_assign_dict.items())
        clone_assign_df.rename(columns={0: "cell_id", 1: "clone_id"}, inplace=True)
        gene_type_score_df = pd.DataFrame.from_dict(self.gene_type_score_dict.items())
        gene_type_score_df.rename(columns={0: "gene", 1: "gene_type_score"}, inplace=True)

        return clone_assign_df, gene_type_score_df

    def __init__(self, expr, cnv, normalize_cnv=True, cnv_cutoff=10, model_select="gene", repeat=10,
                 min_clone_assign_prob=0.8, min_clone_assign_freq=0.7, min_consensus_gene_freq=0.8,
                 max_temp=1.0, min_temp=0.5, anneal_rate=0.01, learning_rate=0.1, max_iter=400, rel_tol=5e-5):
        '''
        initialize CloneAlign object
        :param expr: expr read count matrix. row is gene, column is cell. (pandas.DataFrame)
        :param cnv: cnv matrix. row is gene, column is cell. (pandas.DataFrame)
        :param normalize_cnv: whether to normalized cnv matrix by min or not. (bool)
        :param cnv_cutoff: set cnv higher than cnv_cutoff to cnv_cutoff. (int)
        :param model_select: "gene" for the extended clonealign model or "default" for the original clonelign model (str)
        :param repeat: num of times to run clonealign to generate consensus results. (int)
        :param min_clone_assign_prob: assign cells to a clone if clone assignment prob reaches min_clone_assign_prob (float)
        :param min_clone_assign_freq: assign cells to a clone if a min proportion of runs generate the same results (float)
        :param max_temp: starting temperature in Gumbel-Softmax reparameterization. (float)
        :param min_temp: min temperature in Gumbel-Softmax reparameterization. (float)
        :param anneal_rate: annealing rate in Gumbel-Softmax reparameterization. (float)
        :param learning_rate: learning rate of Adam optimizer. (float)
        :param max_iter: max number of iterations of elbo optimization during inference. (int)
        :param rel_tol: when the relative change in elbo drops to rel_tol, stop inference. (float)
        '''
        self.map_estimates = None
        self.expr_df = expr
        self.cnv_df = cnv
        self.normalize_cnv = normalize_cnv
        self.cnv_cutoff = cnv_cutoff

        self.min_consensus_gene_freq = min_consensus_gene_freq
        self.model_select = model_select
        self.repeat = repeat

        self.min_clone_assign_prob = min_clone_assign_prob
        self.min_clone_assign_freq = min_clone_assign_freq

        self.max_temp = max_temp
        self.min_temp = min_temp
        self.anneal_rate = anneal_rate
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self.rel_tol = rel_tol

        # clean input matrix
        self.process_input_matrices()

        # output
        self.clone_assign_dict = dict()
        self.gene_type_score_dict = dict()

    @config_enumerate
    def clonealign_pyro_gene_model(self, cnv, expr, temperature=0.5):
        '''
        extended clonealign model
        :param cnv: torch.tensor
        :param expr: torch.tensor
        :param temperature: float
        :return: None
        '''
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
                                        dist.Normal(inverse_softplus(per_copy_expr_guess), torch.ones(num_of_genes) * 20))

            per_copy_expr = softplus(per_copy_expr)


            # draw mean_expr from another softplus-transformed Normal distribution
            mean_expr = pyro.sample('expose_mean_expr',
                                    dist.Normal(inverse_softplus(per_copy_expr_guess), torch.ones(num_of_genes)))

            mean_expr = softplus(mean_expr)


            # draw w from Normal
            w = pyro.sample('expose_w', dist.Normal(torch.zeros(6), torch.sqrt(chi)).to_event(1))

            # sample the gene_type_score from uniform distribution.
            # the score reflects how much the copy number influence expression.
            gene_type_score = pyro.sample('expose_gene_type_score', dist.Dirichlet(torch.ones(2) * 0.1))
            #gene_type_score = gene_type_score.clamp(min=1e-10)

            gene_type = pyro.sample('expose_gene_type',
                                    dist.RelaxedOneHotCategorical(temperature=torch.tensor(temperature),
                                                                  probs=gene_type_score))
            #gene_type = gene_type.clamp(min=1e-10)

        with pyro.plate('cell', num_of_cells):
            # draw clone_assign_prob from Dir
            clone_assign_prob = pyro.sample('expose_clone_assign_prob', dist.Dirichlet(torch.ones(num_of_clones) * 0.1))
            # draw clone_assign from Cat
            clone_assign = pyro.sample('clone_assign', dist.Categorical(clone_assign_prob))

            # draw psi from Normal
            psi = pyro.sample('expose_psi', dist.Normal(torch.zeros(6), torch.ones(6)).to_event(1))

            # construct expected_expr
            expected_expr = (per_copy_expr * Vindex(cnv)[clone_assign] * gene_type[:, 0] +
                             per_copy_expr * gene_type[:, 1]) * \
                            torch.exp(torch.matmul(psi, torch.transpose(w, 0, 1)))

            # draw expr from Multinomial
            pyro.sample('obs', dist.Multinomial(total_count=3000, probs=expected_expr, validate_args=False), obs=expr)

    @config_enumerate
    def clonealign_pyro_model(self, cnv, expr, temperature=0.5):
        '''
        original clonealign model
        :param cnv: torch.tensor
        :param expr: torch.tensor
        :param temperature: float
        :return: None
        '''
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
                                        dist.Normal(inverse_softplus(per_copy_expr_guess), torch.ones(num_of_genes) * 20))
            per_copy_expr = softplus(per_copy_expr)

            # draw w from Normal
            w = pyro.sample('expose_w', dist.Normal(torch.zeros(6), torch.sqrt(chi)).to_event(1))

        with pyro.plate('cell', num_of_cells):
            # draw clone_assign_prob from Dir
            clone_assign_prob = pyro.sample('expose_clone_assign_prob', dist.Dirichlet(torch.ones(num_of_clones) * 0.1))
            # draw clone_assign from Cat
            clone_assign = pyro.sample('clone_assign', dist.Categorical(clone_assign_prob))

            # draw psi from Normal
            psi = pyro.sample('expose_psi', dist.Normal(torch.zeros(6), torch.ones(6)).to_event(1))

            # construct expected_expr
            expected_expr = per_copy_expr * Vindex(cnv)[clone_assign] * torch.exp(
                torch.matmul(psi, torch.transpose(w, 0, 1)))

            # draw expr from Multinomial
            pyro.sample('obs', dist.Multinomial(total_count=3000, probs=expected_expr, validate_args=False), obs=expr)

    def run_clonealign_pyro(self, cnv, expr):
        '''
        clonealign inference
        :param cnv: clone-specific consensus copy number matrix. row is clone. column is gene. (torch.tensor)
        :param expr: gene expression count matrix. row is cell. column is gene. (torch.tensor)
        :return: clone assignment, gene_type_score
        '''
        np_temp = self.max_temp
        losses = []

        optim = pyro.optim.Adam({'lr': self.learning_rate, 'betas': [0.8, 0.99]})
        elbo = TraceEnum_ELBO(max_plate_nesting=1)

        if self.model_select == "gene":
            model = self.clonealign_pyro_gene_model
        else:
            model = self.clonealign_pyro_model

        pyro.clear_param_store()

        global_guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("expose_")))

        svi = SVI(model, global_guide, optim, loss=elbo)

        # start inference
        print('Start Inference.')
        for i in range(self.max_iter):
            if i % 100 == 1:
                np_temp = np.maximum(self.max_temp * np.exp(-self.anneal_rate * i), self.min_temp)
            loss = svi.step(cnv, expr, np_temp)

            if i >= 1:
                loss_diff = abs((losses[-1] - loss) / losses[-1])
                if loss_diff < self.rel_tol:
                    print('ELBO converged at iteration ' + str(i))
                    break

            losses.append(loss)

        map_estimates = global_guide(cnv, expr)

        clone_assign_prob = map_estimates['expose_clone_assign_prob']
        gene_type_score_df = None

        if self.model_select == "gene":
            gene_type_score = map_estimates['expose_gene_type_score']
            gene_type_score_df = pd.DataFrame(gene_type_score.data.numpy())

        clone_assign_prob_df = pd.DataFrame(clone_assign_prob.data.numpy())

        # also record other parameters
        self.map_estimates = map_estimates
        # softplus = Softplus()
        # per_copy_expr = torch.mean(torch.sum(expr, 1)) * softplus(map_estimates['expose_per_copy_expr']) / 3000
        # self.per_copy_expr_df = pd.DataFrame(per_copy_expr.data.numpy())

        return clone_assign_prob_df, gene_type_score_df

    def run_clonealign_pyro_repeat(self, cnv_df, expr_df):
        '''
        call run_clonealign_pyro() multiple times to generate consensus results
        :param cnv_df: clone-specific consensus copy number matrix. row is gene. column is clone. (pandas.DataFrame)
        :param expr_df: gene expression count matrix. row is gene. column is cell. (pandas.DataFrame)
        :return: frequency of unassigned cells, clone assignment, gene_type_score
        '''
        torch.set_default_dtype(torch.float64)
        cnv = torch.tensor(cnv_df.values, dtype=torch.float64)
        cnv = torch.transpose(cnv, 0, 1)

        expr = torch.tensor(expr_df.values, dtype=torch.float64)
        expr = torch.transpose(expr, 0, 1)

        clone_assign_list = []
        gene_type_score_list = []

        for i in range(self.repeat):
            current_clone_assign, gene_type_score = self.run_clonealign_pyro(cnv, expr)

            current_clone_assign_prob = current_clone_assign.max(1)
            current_clone_assign_discrete = current_clone_assign.idxmax(1)

            current_clone_assign_discrete[current_clone_assign_prob < self.min_clone_assign_prob] = np.nan
            clone_assign_list.append(current_clone_assign_discrete)

            if self.model_select == "gene":
                gene_type_score_list.append(gene_type_score.iloc[:, 0])

        clone_assign = pd.concat(clone_assign_list, axis=1)
        if len(gene_type_score_list) > 0:
            gene_type_score = pd.concat(gene_type_score_list, axis=1)
            gene_type_score_mean = gene_type_score.mean(1)
        else:
            gene_type_score = None
            gene_type_score_mean = None

        clone_assign_max = clone_assign.mode(1, dropna=False)[0]
        clone_assign_freq = clone_assign.apply(max_count, axis=1) / self.repeat

        clone_assign_max[clone_assign_freq < self.min_clone_assign_freq] = np.nan

        # calculate NaN freq
        none_freq = clone_assign_max.isna().sum() / clone_assign_max.shape[0]

        return none_freq, clone_assign_max, gene_type_score_mean, clone_assign, gene_type_score
