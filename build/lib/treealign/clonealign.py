"""
CloneAlign base class: initialize and clean expr, cnv data frame
"""

import torch
import time
import numpy as np
import pandas as pd
from torch.nn import Softplus

import pyro
import pyro.optim
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.infer.autoguide.initialization import init_to_sample
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
        contains_total_copy_data = self.cnv_df is not None and self.expr_df is not None
        contains_allele_specific_data = self.hscn_df is not None and self.snv_allele_df is not None and self.snv_df is not None
        
        if (not contains_total_copy_data) and (not contains_allele_specific_data):
            raise ValueError('Both total copy number data and allele specific data are missing!')
        
        if contains_total_copy_data:
            self.cnv_df = self.cnv_df[self.cnv_df.var(1) > 0]
            self.expr_df = self.expr_df[self.expr_df.mean(1) > 0]

            # normalize expr_df by total read count per cell
            expr_total_count = self.expr_df.sum(axis=0)
            # expr_total_count_median = pd.median(expr_total_count)            
            self.expr_df = self.expr_df * 5000
            self.expr_df = self.expr_df.div(expr_total_count, axis="columns")
    
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
            
            # add offsets to 0 and 1 in baf matrix
            self.hscn_df[self.hscn_df == 0] = 0.1
            self.hscn_df[self.hscn_df == 1] = 0.9
        
        if contains_total_copy_data and contains_allele_specific_data:
            intersect_cells = self.expr_df.columns & self.snv_allele_df.columns & self.snv_df.columns
            
            self.expr_df = self.expr_df[intersect_cells]
            self.snv_allele_df = self.snv_allele_df[intersect_cells]
            self.snv_df = self.snv_df[intersect_cells]
        
        return
    
    def construct_total_copy_number_input(self, terminals, expr_cells):
        if self.cnv_df is None or self.expr_df is None:
            return None, None
        # get clone specific cnv profiles
        clone_cnv_list = []
        mode_freq_list = []
        for terminal in terminals:
            clean_terminal = [t for t in terminal if t in self.cnv_df.columns]
            if len(clean_terminal) == 0:
                print("Too many cells in the phylogenetic tree don't have copy number profiles. Please double check you are using the correct CN matrix.")
                return None, None
            cnv_subset = self.cnv_df[clean_terminal]
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
        
        clone_cnv_df_dedup = clone_cnv_df[~clone_cnv_df.index.duplicated(keep='first')]
        
        return expr_input, clone_cnv_df_dedup
    
    
    def construct_allele_specific_input(self, terminals, expr_cells):
        if self.hscn_df is None or self.snv_allele_df is None or self.snv_df is None:
            return None, None, None
          
        clone_hscn_list = []
        mode_freq_list = []
        
        for terminal in terminals:
            clean_terminal = [t for t in terminal if t in self.hscn_df.columns]
            if len(clean_terminal) == 0:
                print("Too many cells in the phylogenetic tree don't have BAF profiles. Please double check you are using the correct BAF input.")
                return None, None, None
            hscn_subset = self.hscn_df[clean_terminal]
            current_mode = hscn_subset.mode(1)[0]
            clone_hscn_list.append(current_mode)
            mode_freq_list.append(hscn_subset.eq(current_mode, axis=0).sum(axis=1).div(hscn_subset.shape[1]))
        
        clone_hscn_df = pd.concat(clone_hscn_list, axis=1)
        mode_freq_df = pd.concat(mode_freq_list, axis=1)
        
        variance_filter = clone_hscn_df.var(1).gt(0)
        mode_freq_filter = mode_freq_df.min(axis=1).gt(self.min_consensus_snv_freq)
        clone_hscn_df = clone_hscn_df[variance_filter & mode_freq_filter]
        
        snv = self.snv_df[expr_cells]
        snv_allele = self.snv_allele_df[expr_cells]
        
        intersect_index = clone_hscn_df.index & snv.index & snv_allele.index
        
        clone_hscn_df = clone_hscn_df.loc[intersect_index, ]
        snv = snv.loc[intersect_index, ]
        snv_allele = snv_allele.loc[intersect_index, ]
        
        return clone_hscn_df, snv_allele, snv
    
    # inplace method
    def average_param_dict(self, param_dict):
        if param_dict is not None:
            for key in param_dict:
                param_dict[key] = sum(param_dict[key])/len(param_dict[key])
    
    # inplace method            
    def max_param_dict(self, param_dict):
        if param_dict is not None:
            for key in param_dict:
                param_dict[key] = max(param_dict[key])
    
    # inplace method
    def make_columns_consistent(self, *args):
        intersect_columns = None
        for arg in args:
            if hasattr(arg, 'columns'):
                if intersect_columns is None:
                    intersect_columns = arg.columns
                else:
                    intersect_columns = intersect_columns.intersection(arg.columns)
        for arg in args:
            if hasattr(arg, 'columns'):
                arg.drop(columns=[col for col in arg if col not in intersect_columns], inplace=True)
                
    def convert_df_to_torch(self, df):
        if df is not None:
            df_torch = torch.tensor(df.values, dtype=torch.float64)
            df_torch = torch.transpose(df_torch, 0, 1)
            return df_torch
        else:
            return None

    def generate_output(self):
        '''
        generate clone_assign_df and gene_type_score_df
        :return: clonealign output (pandas.DataFrame)
        '''
        if self.clone_assign_df is None:
            self.clone_assign_df = pd.DataFrame.from_dict(self.clone_assign_dict.items())
            self.clone_assign_df.rename(columns={0: "cell_id", 1: "clone_id"}, inplace=True)
        if self.gene_type_score_df is None:
            if self.gene_type_score_dict is None:
                self.gene_type_score_df = None
            else:
                self.average_param_dict(self.gene_type_score_dict)
                self.gene_type_score_df = pd.DataFrame.from_dict(self.gene_type_score_dict.items())
                self.gene_type_score_df.rename(columns={0: "gene", 1: "gene_type_score"}, inplace=True)
        
        if self.allele_assign_prob_df is None:
            if self.allele_assign_prob_dict is None:
                self.allele_assign_prob_df = None
            else:
                self.average_param_dict(self.allele_assign_prob_dict)
                self.allele_assign_prob_df = pd.DataFrame.from_dict(self.allele_assign_prob_dict.items())
                self.allele_assign_prob_df.rename(columns={0: "snp", 1: "allele_assign_prob"}, inplace=True)

        return self.clone_assign_df, self.gene_type_score_df, self.allele_assign_prob_df

    def __init__(self, expr=None, cnv=None, hscn=None, snv_allele=None, snv=None, 
                 normalize_cnv=True, cnv_cutoff=10, infer_s_score=True, infer_b_allele=True, repeat=10,
                 min_clone_assign_prob=0.8, min_clone_assign_freq=0.7, min_consensus_gene_freq=0.6,min_consensus_snv_freq=0.6,
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
        self.min_consensus_snv_freq = min_consensus_snv_freq
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
        
        self.clone_assign_df = None
        self.gene_type_score_df = None
        self.allele_assign_prob_df = None

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
        
        has_total_copy_number_data = cnv is not None and expr is not None
        has_allele_specific_data = hscn is not None and snv_allele is not None and snv is not None

        if has_total_copy_number_data:
            num_of_clones = len(cnv)
            num_of_cells = len(expr)
            num_of_genes = len(expr[0])
            
            softplus = Softplus()
            
            #expr_total_count = torch.sum(expr, 1)
            # expr_total_count_median = torch.median(expr_total_count)
            
            # initialize per_copy_expr using the data (This typically speeds up convergence)
            # expr = expr * expr_total_count_median / torch.reshape(expr_total_count, (num_of_cells, 1))
            per_copy_expr_guess = torch.mean(expr, 0)

            # draw chi from gamma
            chi = pyro.sample('expose_chi', dist.Gamma(torch.ones(6) * 2, torch.ones(6)).to_event(1))            

        if has_allele_specific_data:
            num_of_clones = len(hscn)
            num_of_cells = len(snv)
            num_of_snps = len(hscn[0])
            hscn_complement = 1 - hscn


        if has_total_copy_number_data:
            with pyro.plate('gene', num_of_genes):
                # draw per_copy_expr from softplus-transformed Normal distribution
                per_copy_expr = pyro.sample('expose_per_copy_expr',
                                            dist.Normal(inverse_softplus(per_copy_expr_guess), torch.ones(num_of_genes) * 10))
                per_copy_expr = softplus(per_copy_expr)

                # draw w from Normal
                w = pyro.sample('expose_w', dist.Normal(torch.zeros(6), torch.sqrt(chi)).to_event(1))

                if infer_s_score:
                    # sample the gene_type_score from uniform distribution.
                    # the score reflects how much the copy number influence expression.
                    gene_type_score = pyro.sample('expose_gene_type_score', dist.Dirichlet(torch.ones(2) * 1))
                    gene_type = pyro.sample('expose_gene_type', dist.RelaxedOneHotCategorical(temperature=torch.tensor(temperature),
                                                                          probs=gene_type_score))

        # infer whether reference allele is b allele
        if has_allele_specific_data and infer_b_allele:
            with pyro.plate('snp', num_of_snps):
                # draw allele assignment prob from beta dist
                allele_assign_prob = pyro.sample('expose_allele_assign_prob',  dist.Dirichlet(torch.ones(2) * 1))
                allele_assign = pyro.sample('expose_allele', dist.RelaxedOneHotCategorical(temperature=torch.tensor(temperature),
                                                                          probs=allele_assign_prob))        

        with pyro.plate('cell', num_of_cells):
            # draw clone_assign_prob from Dir
            clone_assign_prob = pyro.sample('expose_clone_assign_prob', dist.Dirichlet(torch.ones(num_of_clones) * 1))
            # draw clone_assign from Cat
            clone_assign = pyro.sample('clone_assign', dist.Categorical(clone_assign_prob))

            # construct expected_expr
            if has_total_copy_number_data:
                # draw psi from Normal
                psi = pyro.sample('expose_psi', dist.Normal(torch.zeros(6), torch.ones(6)).to_event(1))                
                if infer_s_score:
                    expected_expr = (per_copy_expr * Vindex(cnv)[clone_assign] * gene_type[:, 0] + per_copy_expr * gene_type[:, 1]) * torch.exp(torch.matmul(psi, torch.transpose(w, 0, 1)))
                else:
                    expected_expr = per_copy_expr * Vindex(cnv)[clone_assign] * torch.exp(torch.matmul(psi, torch.transpose(w, 0, 1)))
                # draw expr from Multinomial
                pyro.sample('cnv', dist.Multinomial(total_count=5000, probs=expected_expr, validate_args=False), obs=expr)

            if has_allele_specific_data:
                if infer_b_allele:
                    real_hscn = Vindex(hscn)[clone_assign] * allele_assign[:, 0] + Vindex(hscn_complement)[clone_assign] * allele_assign[:, 1]
                else:
                    real_hscn = Vindex(hscn)[clone_assign]
                pyro.sample('hscn', dist.Binomial(snv, real_hscn).to_event(1), obs = snv_allele)
                
        
    
    def run_clonealign_pyro(self, cnv, expr, hscn, snv_allele, snv, current_repeat):
        '''
        clonealign inference
        :param cnv: clone-specific consensus copy number matrix. row is clone. column is gene. (torch.tensor)
        :param expr: gene expression count matrix. row is cell. column is gene. (torch.tensor)
        :return: clone assignment, gene_type_score
        '''
        # record start time
        start_time = round(time.time()*1000)

        np_temp = self.max_temp
        losses = []

        optim = pyro.optim.Adam({'lr': self.learning_rate, 'betas': [0.8, 0.99]})
        elbo = TraceEnum_ELBO(max_plate_nesting=1)

        model = self.clonealign_pyro_model
        
        # def initialize(seed):
        #     global global_guide, svi
        #     pyro.set_rng_seed(seed)
        #     pyro.clear_param_store()
        #     global_guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("expose_")), init_loc_fn=init_to_sample)
        #     svi = SVI(model, global_guide, optim, loss=elbo)
        #     return svi.loss(model, global_guide, cnv, expr, hscn, snv_allele, snv, self.infer_s_score, self.infer_b_allele, np_temp)
                  
        # loss, seed = min((initialize(seed), seed) for seed in range(current_repeat * 100, (current_repeat + 1) * 100))
        # initialize(seed)
        pyro.set_rng_seed(current_repeat * 4 + 1)
        pyro.clear_param_store()
        global_guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("expose_")))

        svi = SVI(model, global_guide, optim, loss=elbo)

        #print('seed = {}, initial_loss = {}'.format(seed, loss))
        
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
          results[entry_name] = pd.DataFrame(map_estimates[entry].data.numpy())


        # end time
        end_time = round(time.time()*1000)
        
        # add run time and losses
        results["losses"] = losses
        results["time"] = end_time - start_time
        
        return results

    def run_clonealign_pyro_repeat(self, cnv_df, expr_df, hscn_df, snv_allele_df, snv_df):
        '''
        call run_clonealign_pyro() multiple times to generate consensus results
        :param cnv_df: clone-specific consensus copy number matrix. row is gene. column is clone. (pandas.DataFrame)
        :param expr_df: gene expression count matrix. row is gene. column is cell. (pandas.DataFrame)
        :return: frequency of unassigned cells, clone assignment, gene_type_score
        '''
        torch.set_default_dtype(torch.float64)
        cnv = self.convert_df_to_torch(cnv_df)
        expr = self.convert_df_to_torch(expr_df)
        hscn = self.convert_df_to_torch(hscn_df)
        snv_allele = self.convert_df_to_torch(snv_allele_df)
        snv = self.convert_df_to_torch(snv_df)

        clone_assign_list = []
        other_params = dict()

        losses_dfs = []
        times = {"repeat": [], "time": []}
        for i in range(self.repeat):
            current_results = self.run_clonealign_pyro(cnv, expr, hscn, snv_allele, snv, i)
            
            current_clone_assign = current_results['clone_assign_prob']
            current_clone_assign_prob = current_clone_assign.max(1)
            current_clone_assign_discrete = current_clone_assign.idxmax(1)

            current_clone_assign_discrete[current_clone_assign_prob < self.min_clone_assign_prob] = np.nan
            clone_assign_list.append(current_clone_assign_discrete)
            
            skip_names = ['clone_assign_prob', 'time', 'losses']
            for param_name in current_results:
                if param_name not in skip_names:
                    if param_name not in other_params:
                        other_params[param_name] = []
                    other_params[param_name].append(current_results[param_name].iloc[:, 0])
            
            # parse out run time data
            iter_count = len(current_results["losses"])
            losses = {"repeat": [i for _ in range(iter_count)], "iter": [j for j in range(iter_count)], "loss": current_results["losses"]}
            losses_df = pd.DataFrame(losses)
            losses_dfs.append(losses_df)

            times["repeat"].append(i)
            times["time"].append(current_results["time"])
        
        if len(losses_dfs) != 0:
            losses_result = pd.concat(losses_dfs, ignore_index=True)
        else:
            losses_result = pd.DataFrame()
        times_result = pd.DataFrame(times)

        mean_params = dict()
        for param_name in other_params:
            other_params[param_name] = pd.concat(other_params[param_name], axis=1)
            mean_params['mean_' + param_name] = other_params[param_name].mean(1)
            
        other_params.update(mean_params)
            
        clone_assign = pd.concat(clone_assign_list, axis=1)
        clone_assign_max = clone_assign.mode(1, dropna=False)[0]
        clone_assign_freq = clone_assign.apply(max_count, axis=1) / self.repeat

        clone_assign_max[clone_assign_freq < self.min_clone_assign_freq] = np.nan

        # calculate NaN freq
        none_freq = clone_assign_max.isna().sum() / clone_assign_max.shape[0]

        return none_freq, clone_assign_max, clone_assign, other_params, losses_result, times_result
      
    
