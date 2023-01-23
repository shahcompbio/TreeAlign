"""
CloneAlignClone class
"""

import pandas as pd
import numpy as np
from .clonealign import CloneAlign


class CloneAlignClone(CloneAlign):

    def __init__(self, clone, expr=None, cnv=None, hscn=None, snv_allele=None, snv=None, 
                 normalize_cnv=True, cnv_cutoff=10, infer_s_score=True, infer_b_allele=True, repeat=10,
                 min_clone_assign_prob=0.8, min_clone_assign_freq=0.7, min_consensus_gene_freq=0.6,min_consensus_snv_freq=0.6,
                 max_temp=1.0, min_temp=0.5, anneal_rate=0.01, learning_rate=0.1, max_iter=400, rel_tol=5e-5, 
                 record_input_output=False, 
                 min_clone_cell_count=10):
        '''
        initialize CloneAlignClone object
        :param expr: expr read count matrix. row is gene, column is cell. (pandas.DataFrame)
        :param cnv: cnv matrix. row is gene, column is cell. (pandas.DataFrame)
        :param clone: groupings of cnv cells. (pandas.DataFrame)
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
        CloneAlign.__init__(self, expr, cnv, hscn, snv_allele, snv, 
                            normalize_cnv, cnv_cutoff, infer_s_score, infer_b_allele, 
                            repeat, min_clone_assign_prob, min_clone_assign_freq, min_consensus_snv_freq,
                            min_consensus_gene_freq, max_temp, min_temp, anneal_rate, 
                            learning_rate, max_iter, rel_tol, record_input_output)

        self.clone_df = clone
        self.clone_df.rename(columns={0: "cell_id", 1: "clone_id"})

        clone_cell_counts = self.clone_df['clone_id'].value_counts()
        cells_to_keep = clone_cell_counts[clone_cell_counts >= min_clone_cell_count].index.values

        self.clone_df = self.clone_df[self.clone_df['clone_id'].isin(cells_to_keep)]

        if self.clone_df.shape[1] <= 1:
            raise ValueError('There are less than 2 clones in the input. Add more clones to run CloneAlign.')


    def assign_cells_to_clones(self):
        '''
        assign cells from scRNA to clones identified in scDNA
        :return: clone_assign_df (pandas.DataFrame) and gene_type_score_df (pandas.DataFrame)
        '''
        terminals = []
        if self.expr_df is not None:
            expr_cells = self.expr_df.columns.values
        if self.snv_df is not None:
            expr_cells = self.snv_df.columns.values
        clones = self.clone_df["clone_id"].drop_duplicates().values
        
        for clone in clones:
            terminal = self.clone_df.loc[self.clone_df["clone_id"] == clone, "cell_id"].values
            terminals.append(terminal)
        
        # construct total copy number input
        expr_input, clone_cnv_df = self.construct_total_copy_number_input(terminals, expr_cells)
        clone_cnv_df.columns = clones
        # construct allele specific input
        hscn_input, snv_allele_input, snv_input = self.construct_allele_specific_input(terminals, expr_cells) 
        
        # make columns consistent
        self.make_columns_consistent(expr_input, snv_allele_input, snv_input)
        
        
        has_allele_specific_data = hscn_input is not None and snv_allele_input is not None and snv_input is not None
        has_total_copy_number_data = expr_input is not None and clone_cnv_df is not None        
        
        # run clonealign
        gene_count = 0
        snp_count = 0
        cell_count = 0
        if has_total_copy_number_data:
            gene_count = clone_cnv_df.shape[0]
            cell_count = expr_input.shape[1]
            print("gene count: " + str(gene_count))
        
        if has_allele_specific_data:
            snp_count = hscn_input.shape[0]
            cell_count = snv_input.shape[1]
            print("snp count: " + str(snp_count))
        
        if gene_count == 0 and snp_count == 0:
            raise ValueError('No valid genes or snps exist in the matrix after filtering. Maybe loose the filtering criteria?')
        print("cell count: " + str(cell_count))

        # record input
        if self.record_input_output:
            self.params_dict = dict()
            self.params_dict['input'] = dict()
            self.params_dict['input']['cnv'] = clone_cnv_df
            self.params_dict['input']['expr'] = expr_input
            self.params_dict['input']['hscn'] = hscn_input
            self.params_dict['input']['snv_allele'] = snv_allele_input
            self.params_dict['input']['snv'] = snv_input
        
        none_freq, clone_assign, clone_assign_df, params_dict = self.run_clonealign_pyro_repeat(clone_cnv_df, expr_input, hscn_input, snv_allele_input, snv_input)
        
        if self.record_input_output:
            self.params_dict['output'] = dict()
            self.params_dict['output']['none_freq'] = none_freq
            self.params_dict['output']['clone_assign'] = clone_assign
            self.params_dict['output']['clone_assign_df'] = clone_assign_df
            self.params_dict['output']['params_dict'] = params_dict
        
        
        if has_total_copy_number_data:
            cell_count = expr_input.shape[1]
            cell_names = expr_input.columns.values
        if not has_total_copy_number_data:
            cell_count = snv_input.shape[1]
            cell_names = snv_input.columns.values
        # record clone_assign
        for i in range(cell_count):
            if np.isnan(clone_assign.values[i]):
                self.clone_assign_dict[cell_names[i]] = np.nan
            else:
                self.clone_assign_dict[cell_names[i]] = clones[int(clone_assign.values[i])]
                
                
        # record gene_type_score
        if has_total_copy_number_data and self.infer_s_score:
            for i in range(expr_input.shape[0]):
                self.gene_type_score_dict[expr_input.index.values[i]] = [params_dict['mean_gene_type_score'][i]]
        if has_allele_specific_data and self.infer_b_allele:
            for i in range(hscn_input.shape[0]):
                self.allele_assign_prob_dict[hscn_input.index.values[i]] = [params_dict['mean_allele_assign_prob'][i]]

        return
