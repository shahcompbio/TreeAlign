"""
CloneAlignClone class
"""

import pandas as pd
import numpy as np
from .clonealign import CloneAlign


class CloneAlignClone(CloneAlign):

    def __init__(self, expr, cnv, clone, normalize_cnv=True, cnv_cutoff=10, model_select="gene", repeat=10,
                 min_clone_cell_count=20,
                 min_clone_assign_prob=0.8, min_clone_assign_freq=0.7,min_consensus_gene_freq=0.2,
                 max_temp=1.0, min_temp=0.5, anneal_rate=0.01, learning_rate=0.1, max_iter=400, rel_tol=5e-5):
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
        CloneAlign.__init__(self, expr, cnv, normalize_cnv, cnv_cutoff, model_select, repeat, min_clone_assign_prob,
                            min_clone_assign_freq, min_consensus_gene_freq, max_temp, min_temp, anneal_rate, learning_rate, max_iter, rel_tol)

        self.clone_df = clone
        self.clone_df.rename(columns={0: "cell_id", 1: "clone_id"})
        # only keep cells that have cnv profile
        self.clone_df = self.clone_df[self.clone_df['cell_id'].isin(self.cnv_df.columns.values)]

        clone_cell_counts = self.clone_df['clone_id'].value_counts()
        cells_to_keep = clone_cell_counts[clone_cell_counts >= min_clone_cell_count].index.values

        self.clone_df = self.clone_df[self.clone_df['clone_id'].isin(cells_to_keep)]
        self.clone_cnv_df = None

        if self.clone_df.shape[1] <= 1:
            raise ValueError('There are less than 2 clones in the input. Add more clones to run CloneAlign.')

        self.clone_assign_df = None
        self.gene_type_score_df = None

    def generate_output(self):
        summarized_clone_assign_df, summarized_gene_type_score_df = CloneAlign.generate_output(self)
        return summarized_clone_assign_df, summarized_gene_type_score_df, self.clone_assign_df, self.gene_type_score_df


    def assign_cells_to_clones(self):
        '''
        assign cells from scRNA to clones identified in scDNA
        :return: clone_assign_df (pandas.DataFrame) and gene_type_score_df (pandas.DataFrame)
        '''
        clone_cnv_list = []
        mode_freq_list = []
        clones = self.clone_df["clone_id"].drop_duplicates().values

        for c in clones:
            clone_cells = self.clone_df.loc[self.clone_df["clone_id"] == c, "cell_id"].values
            cnv_subset = self.cnv_df[clone_cells]
            current_mode = cnv_subset.mode(1)[0]
            clone_cnv_list.append(current_mode)
            mode_freq_list.append(cnv_subset.eq(current_mode, axis=0).sum(axis=1).div(cnv_subset.shape[1]))

        clone_cnv_df = pd.concat(clone_cnv_list, axis=1)
        mode_freq_df = pd.concat(mode_freq_list, axis=1)

        clone_cnv_df.columns = clones

        variance_filter = clone_cnv_df.var(1).gt(0)
        mode_freq_filter = mode_freq_df.min(axis=1).gt(self.min_consensus_gene_freq)
        clone_cnv_df = clone_cnv_df[variance_filter & mode_freq_filter]
        self.cnv_df = self.cnv_df[variance_filter & mode_freq_filter]
        # normalize cnv
        if self.normalize_cnv:
            cnv_correction = clone_cnv_df[clone_cnv_df > 0].min(axis=1)
            clone_cnv_df = clone_cnv_df.div(cnv_correction, axis=0)
            self.cnv_df = self.cnv_df.div(cnv_correction, axis=0)

        expr_input = self.expr_df[self.expr_df.mean(1) > 0]

        intersect_index = clone_cnv_df.index.intersection(expr_input.index)

        expr_input = expr_input.loc[intersect_index,]
        clone_cnv_df = clone_cnv_df.loc[intersect_index,]

        # reorder self.cnv_df and self.expr_df
        self.expr_df = self.expr_df.loc[intersect_index, ]
        self.cnv_df = self.cnv_df.loc[intersect_index, ]

        # run clonealign
        clone_count = clone_cnv_df.shape[1]

        self.clone_cnv_df = clone_cnv_df
        print(f'Start run clonealign for {clone_count} clones:')
        print("cnv gene count: " + str(clone_cnv_df.shape[0]))
        print("expr cell count: " + str(expr_input.shape[1]))

        none_freq, clone_assign, gene_type_score, self.clone_assign_df, self.gene_type_score_df = self.run_clonealign_pyro_repeat(clone_cnv_df, expr_input)

        clones_dict = dict()
        for i in range(len(clones)):
            clones_dict[float(i)] = clones[i]

        self.clone_assign_df.replace(clones_dict, inplace=True)
        self.clone_assign_df.index = expr_input.columns.values
        if self.gene_type_score_df is not None:
            self.gene_type_score_df.index = expr_input.index.values

        # record clone_assign
        for i in range(expr_input.shape[1]):
            if np.isnan(clone_assign.values[i]):
                self.clone_assign_dict[expr_input.columns.values[i]] = np.nan
            else:
                self.clone_assign_dict[expr_input.columns.values[i]] = clones[int(clone_assign.values[i])]

        # record gene_type_score
        if gene_type_score is not None:
            for i in range(expr_input.shape[0]):
                self.gene_type_score_dict[expr_input.index.values[i]] = gene_type_score.values[i]

        return self.generate_output()
