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
        contains_total_copy_data = self.cnv_df != None or self.expr_df != None
        contains_allele_specific_data = self.hscn_df != None or self.snv_allele_df != None or self.snv_df != None
        
        if contains_total_copy_data:
            self.cnv_df = self.cnv_df[self.cnv_df.var(1) > 0]
            self.expr_df = self.expr_df[self.expr_df.mean(1) > 0]
    
            intersect_index = self.cnv_df.index.intersection(self.expr_df.index)
    
            self.expr_df = self.expr_df.loc[intersect_index,]
            self.cnv_df = self.cnv_df.loc[intersect_index,]
    
            self.cnv_df[self.cnv_df > self.cnv_cutoff] = self.cnv_cutoff
        
        if contains_allele_specific_data:
            intersect_index = self.hscn_df.index & self.snv_allele_df.index & self.snv_df.index
            
            intersect_cells = self.snv_allele_df.columns & self.snv_df.columns
            
            self.hscn_df = self.hscn_df.loc[intersect_index, ]
            self.snv_allele_df = self.snv_allele_df.loc[intersect_index, intersect_cells]
            self.snv_df = self.snv_df.loc[intersect_index, intersect_cells]
        
        if contains_total_copy_data and contains_allele_specific_data:
            intersect_cells = self.expr_df.columns & self.snv_allele_df.columns & self.snv_df.columns
            
            self.expr_df = self.expr_df.loc[, intersect_cells]
            self.snv_allele_df = self.snv_allele_df.loc[, intersect_cells]
            self.snv_df = self.snv_df.loc[, intersect_cells]
        
        return
    
    def construct_total_copy_number_input(self, terminals, expr_cells):
        # get clone specific cnv profiles
        clone_cnv_list = []
        mode_freq_list = []
        for terminal in terminals:
            cnv_subset = self.cnv_df[terminal]
            current_mode = cnv_subset.mode(1)[0]
            clone_cnv_list.append(current_mode)
            mode_freq_list.append(cnv_subset.eq(current_mode, axis=0).sum(axis=1).div(cnv_subset.shape[1]))

        # concatenate clone_cnv_list
        clone_cnv_df = pd.concat(clone_cnv_list, axis=1)
        mode_freq_df = pd.concat(mode_freq_list, axis=1)

        # remove non-variable genes
        variance_filter = clone_cnv_df.var(1).gt(0)
        mode_freq_filter = mode_freq_df.min(axis=1).gt(self.min_consensus_gene_freq)
        clone_cnv_df = clone_cnv_df[variance_filter & mode_freq_filter]
        # cnv normalization
        if self.normalize_cnv:
            clone_cnv_df = clone_cnv_df.div(clone_cnv_df[clone_cnv_df > 0].min(axis=1), axis=0)


        expr_input = self.expr_df[expr_cells]
        expr_input = expr_input[expr_input.mean(1) > 0]

        intersect_index = clone_cnv_df.index.intersection(expr_input.index)

        expr_input = expr_input.loc[intersect_index,]
        clone_cnv_df = clone_cnv_df.loc[intersect_index,]
        
        return expr_input, clone_cnv_df
    
    
    def construct_allele_specific_input(self, terminals, expr_cells):
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

    def __init__(self, expr=None, cnv=None, hscn=None, snv_allele=None, snv=None, 
                 normalize_cnv=True, cnv_cutoff=10, infer_s_score=True, infer_b_allele=True, repeat=10,
                 min_clone_assign_prob=0.8, min_clone_assign_freq=0.7, min_consensus_gene_freq=0.8,
                 max_temp=1.0, min_temp=0.5, anneal_rate=0.01, learning_rate=0.1, max_iter=400, rel_tol=5e-5, 
                 record_input_output=False):
        '''
        initialize CloneAlign object
        :param expr: expr read count matrix. row is gene, column is cell. (pandas.DataFrame)
        :param cnv: cnv matrix. row is gene, column is cell. (pandas.DataFrame)
        :param normalize_cnv: whether to normalized cnv matrix by min or not. (bool)
        :param cnv_cutoff: set cnv higher than cnv_cutoff to cnv_cutoff. (int)
        :param infer_s_score: infer correlation between copy number and expression (bool)
        :param infer_b_allele: infer whether ref allele is b allele
        :param repeat: num of times to run clonealign to generate consensus results. (int)
        :param min_clone_assign_prob: assign cells to a clone if clone assignment prob reaches min_clone_assign_prob (float)
        :param min_clone_assign_freq: assign cells to a clone if a min proportion of runs generate the same results (float)
        :param max_temp: starting temperature in Gumbel-Softmax reparameterization. (float)
        :param min_temp: min temperature in Gumbel-Softmax reparameterization. (float)
        :param anneal_rate: annealing rate in Gumbel-Softmax reparameterization. (float)
        :param learning_rate: learning rate of Adam optimizer. (float)
        :param max_iter: max number of iterations of elbo optimization during inference. (int)
        :param rel_tol: when the relative change in elbo drops to rel_tol, stop inference. (float)
        :param record_input_output: record input output before and after clonealign runs. (bool)
        '''
        self.map_estimates = None
        self.expr_df = expr
        self.cnv_df = cnv
        self.hscn_df = hscn
        self.snv_allele_df = snv_allele
        self.snv_df = snv
        self.normalize_cnv = normalize_cnv
        self.cnv_cutoff = cnv_cutoff

        self.min_consensus_gene_freq = min_consensus_gene_freq
        self.infer_s_score = infer_s_score
        self.infer_b_allele = infer_b_allele
        self.repeat = repeat

        self.min_clone_assign_prob = min_clone_assign_prob
        self.min_clone_assign_freq = min_clone_assign_freq

        self.max_temp = max_temp
        self.min_temp = min_temp
        self.anneal_rate = anneal_rate
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.record_input_output = record_input_output

        self.rel_tol = rel_tol

        # clean input matrix
        self.process_input_matrices()

        # output
        self.clone_assign_dict = dict()
        self.gene_type_score_dict = dict()
        self.allele_assign_prob_dict = dict()
        self.params_dict = dict()

    @config_enumerate
    def clonealign_pyro_model(self, cnv, expr, hscn, snv_allele, snv, infer_s_score=True, infer_b_allele=True, temperature=0.5):
        '''
        original clonealign model
        :param cnv: torch.tensor
        :param expr: torch.tensor
        :param hscn:  torch.tensor
        :param snv_allele: torch.tensor
        :param snv: torch.tensor
        :param gene_model: bool
        :param infer_b_allele: bool
        :param temperature: float
        :return: None
        '''

        num_of_clones = 0
        num_of_snps = 0
        num_of_cells = 0
        num_of_genes = 0
        hscn_complement = None
               
        if cnv != None and expr != None:
            num_of_clones = len(cnv)
            num_of_cells = len(expr)
            num_of_genes = len(expr[0])

        if hscn != None and snv_allele != None and snv != None:
            num_of_snps = len(hscn[0])
            hscn_complement = 1 - hscn


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

            # draw w from Normal
            w = pyro.sample('expose_w', dist.Normal(torch.zeros(6), torch.sqrt(chi)).to_event(1))

            if infer_s_score:
                # sample the gene_type_score from uniform distribution.
                # the score reflects how much the copy number influence expression.
                gene_type_score = pyro.sample('expose_gene_type_score', dist.Dirichlet(torch.ones(2) * 0.1))
                gene_type = pyro.sample('gene_type',
                                        dist.RelaxedOneHotCategorical(temperature=torch.tensor(temperature),
                                                                      probs=gene_type_score))

        # infer whether reference allele is b allele
        if infer_b_allele:
            with pyro.plate('snp', num_of_snps):
                # draw allele assignment prob from beta dist
                allele_assign_prob = pyro.sample('expose_allele_assign_prob',  dist.Dirichlet(torch.ones(2) * 1))
                allele_assign = pyro.sample('allele', dist.RelaxedOneHotCategorical(temperature=torch.tensor(temperature),
                                                                          probs=allele_assign_prob))        

        with pyro.plate('cell', num_of_cells):
            # draw clone_assign_prob from Dir
            clone_assign_prob = pyro.sample('expose_clone_assign_prob', dist.Dirichlet(torch.ones(num_of_clones) * 0.1))
            # draw clone_assign from Cat
            clone_assign = pyro.sample('clone_assign', dist.Categorical(clone_assign_prob))

            # construct expected_expr
            if cnv != None and expr != None:
                # draw psi from Normal
                psi = pyro.sample('expose_psi', dist.Normal(torch.zeros(6), torch.ones(6)).to_event(1))                
                if infer_s_score:
                    expected_expr = (per_copy_expr * Vindex(cnv)[clone_assign] * gene_type[:, 0] + per_copy_expr * gene_type[:, 1]) * 
                    torch.exp(torch.matmul(psi, torch.transpose(w, 0, 1)))
                else:
                    expected_expr = per_copy_expr * Vindex(cnv)[clone_assign] * torch.exp(torch.matmul(psi, torch.transpose(w, 0, 1)))
                # draw expr from Multinomial
                pyro.sample('cnv', dist.Multinomial(total_count=3000, probs=expected_expr, validate_args=False), obs=expr)

            if hscn != None and snv_allele != None and snv != None:
                if infer_b_allele:
                    real_hscn = Vindex(hscn)[clone_assign] * allele_assign[:, 0] + Vindex(hscn_complement)[clone_assign] * allele_assign[:, 1]
                else:
                    real_hscn = hscn
                pyro.sample('hscn', dist.Binomial(snv, real_hscn).to_event(1), obs = snv_allele)

    def run_clonealign_pyro(self, cnv, expr, hscn, snv_allele, snv):
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

        model = self.clonealign_pyro_model

        pyro.clear_param_store()

        global_guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("expose_")))

        svi = SVI(model, global_guide, optim, loss=elbo)

        # start inference
        print('Start Inference.')
        for i in range(self.max_iter):
            if i % 100 == 1:
                np_temp = np.maximum(self.max_temp * np.exp(-self.anneal_rate * i), self.min_temp)
            loss = svi.step(cnv, expr, hscn, snv_allele, snv, self.infer_s_score, self.infer_b_allele, np_temp)

            if i >= 1:
                loss_diff = abs((losses[-1] - loss) / losses[-1])
                if loss_diff < self.rel_tol:
                    print('ELBO converged at iteration ' + str(i))
                    break

            losses.append(loss)

        map_estimates = global_guide(cnv, expr, hscn, snv_allele, snv, self.infer_s_score, self.infer_b_allele)

        # also record inferred parameters
        self.map_estimates = map_estimates
        results = dict()
        for entry in map_estimates:
          entry_name = entry[7:]
          results[entry_name] = pd.DataFrame(map_estimates[entry])
    
        return results

    def run_clonealign_pyro_repeat(self, cnv_df, expr_df, hscn_df, snv_allele_df, snv_df):
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
        
        hscn = torch.tensor(hscn_df.values, dtype=torch.float64)
        hscn = torch.transpose(hscn)
        
        snv_allele = torch.tensor(snv_allele_df.values, dtype=torch.float64)
        snv_allele = torch.transpose(snv_allele)
        
        snv = torch.tensor(snv_df.values, dtype=torch.float64)
        snv = torch.transpose(snv)

        clone_assign_list = []
        other_params = dict()

        for i in range(self.repeat):
            current_results = self.run_clonealign_pyro(cnv, expr, hscn, snv_allele, snv)
            
            current_clone_assign = current_results['clone_assign_prob']
            current_clone_assign_prob = current_clone_assign.max(1)
            current_clone_assign_discrete = current_clone_assign.idxmax(1)

            current_clone_assign_discrete[current_clone_assign_prob < self.min_clone_assign_prob] = np.nan
            clone_assign_list.append(current_clone_assign_discrete)
            
            for param_name in current_results:
              if param_name != 'clone_assign_prob':
                if param_name not in other_params:
                  other_params[param_name] = []
                other_params[param_name].append(current_results[param_name].iloc[:, 0])
                  
        
        for param_name in other_params:
            other_params[param_name] = pd.concat(other_params[param_name], axix=1)
            other_params['mean_' + param_name] = other_params[param_name].mean(1)
            
        clone_assign = pd.concat(clone_assign_list, axis=1)
        clone_assign_max = clone_assign.mode(1, dropna=False)[0]
        clone_assign_freq = clone_assign.apply(max_count, axis=1) / self.repeat

        clone_assign_max[clone_assign_freq < self.min_clone_assign_freq] = np.nan

        # calculate NaN freq
        none_freq = clone_assign_max.isna().sum() / clone_assign_max.shape[0]

        return none_freq, clone_assign_max, clone_assign, other_params
      
    
