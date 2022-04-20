"""
CloneAlignTree class
"""
from Bio import Phylo
import pandas as pd
import numpy as np
from .clonealign import CloneAlign


class CloneAlignTree(CloneAlign):

    def __init__(self, tree, expr=None, cnv=None, hscn=None, snv_allele=None, snv=None, 
                 normalize_cnv=True, cnv_cutoff=10, infer_s_score=True, infer_b_allele=True, repeat=10,
                 min_clone_assign_prob=0.8, min_clone_assign_freq=0.7, min_consensus_gene_freq=0.8,
                 max_temp=1.0, min_temp=0.5, anneal_rate=0.01, learning_rate=0.1, max_iter=400, rel_tol=5e-5, 
                 record_input_output=False,
                 min_cell_count_expr=20, min_cell_count_cnv=20, min_gene_diff=100, min_snp_diff=100, level_cutoff=10,
                 min_proceed_freq=0.7):
        '''
        initialize CloneAlignTree object
        :param expr: expr read count matrix. row is gene, column is cell. (pandas.DataFrame)
        :param cnv: cnv matrix. row is gene, column is cell. (pandas.DataFrame)
        :param tree: phylogenetic tree of cells (Bio.Phylo.BaseTree.Tree)
        :param normalize_cnv: whether to normalized cnv matrix by min or not. (bool)
        :param cnv_cutoff: set cnv higher than cnv_cutoff to cnv_cutoff. (int)
        :param model_select: "gene" for the extended clonealign model or "default" for the original clonelign model (str)
        :param repeat: num of times to run clonealign to generate consensus results. (int)
        :param min_cell_count_expr: min cells in scRNA to keep assigning cells to subtrees
        :param min_cell_count_cnv: min cells in the current subtree to proceed to the next level
        :param min_gene_diff: min number of genes that are different between subtrees to keep assigning
        :param level_cutoff: stop clonealign when get to subtrees of certain level
        :param min_proceed_freq: proceed clonealign to the next level if a certain frequency of cells have consistent assignments between runs
        :param min_clone_assign_prob: assign cells to a clone if clone assignment prob reaches min_clone_assign_prob (float)
        :param min_clone_assign_freq: assign cells to a clone if a min proportion of runs generate the same results (float)
        :param max_temp: starting temperature in Gumbel-Softmax reparameterization. (float)
        :param min_temp: min temperature in Gumbel-Softmax reparameterization. (float)
        :param anneal_rate: annealing rate in Gumbel-Softmax reparameterization. (float)
        :param learning_rate: learning rate of Adam optimizer. (float)
        :param max_iter: max number of iterations of elbo optimization during inference. (int)
        :param rel_tol: when the relative change in elbo drops to rel_tol, stop inference. (float)
        '''
        CloneAlign.__init__(self, tree, expr, cnv, hscn, snv_allele, snv, 
                            normalize_cnv, cnv_cutoff, infer_s_score, infer_b_allele, 
                            repeat, min_clone_assign_prob, min_clone_assign_freq, 
                            min_consensus_gene_freq, max_temp, min_temp, anneal_rate, 
                            learning_rate, max_iter, rel_tol, record_input_output)

        self.tree = tree
        self.tree.ladderize()
        self.count = 0
        # add name for nodes if the nodes don't have name
        self.add_tree_node_name(self.tree.clade)

        self.min_cell_count_expr = min_cell_count_expr
        self.min_cell_count_cnv = min_cell_count_cnv
        self.min_gene_diff = min_gene_diff
        self.min_snp_diff = min_snp_diff
        self.level_cutoff = level_cutoff
        self.min_proceed_freq = min_proceed_freq

        # output
        self.pruned_clades = set()

    def add_tree_node_name(self, node):
        if node.is_terminal():
            return
        if node.name is None:
            node.name = "node_" + str(self.count)
            self.count += 1
        for child in node.clades:
            self.add_tree_node_name(child)
        return

    def record_clone_assign_to_dict(self, expr_cells, clone_assign, clean_clades):
        '''
        record clone assignment results to self.clone_assign_dict
        :param expr_cells: cells in expr matrix (list[str])
        :param clone_assign: clone assignments (pandas.Series)
        :param clean_clades: clean clades in the current run (list[Clade])
        :return: None
        '''
        for i in range(len(expr_cells)):
            if not np.isnan(clone_assign[i]):
                self.clone_assign_dict[expr_cells[i]] = clean_clades[int(clone_assign[i])].name

    def record_param_to_dict(self, param_dict, indices, params):
        '''
        Update params in params
        :param indices: gene names (list[str])
        :param gene_type_score: mean gene_type_score across runs (pandas.Series)
        :return:
        '''
        if params is None:
            return
        for i in range(params.shape[0]):
            if indices[i] not in param_dict:
                param_dict[indices[i]] = []
            param_dict[indices[i]] = params[i]
    
    
    def assign_cells_to_tree(self):
        '''
        assign cells to Phylo tree
        :return: clone_assign_df (pandas.DataFrame) and gene_type_score_df (pandas.DataFrame)
        '''
        # output
        self.clone_assign_dict = dict()
        self.gene_type_score_dict = dict()
        self.pruned_clades = set()

        self.assign_cells_to_clade(self.tree.clade, list(self.expr_df.columns), 0)

        return self.generate_output()
      

    def assign_cells_to_clade(self, current_clade, expr_cells, level):
        '''
        assign cells to a clade in Phylo tree
        :param current_clade: (Bio.Phylo.BaseTree.Clade)
        :param expr_cells: cells from scRNA (list[str])
        :param level: current level of the clade
        :return: None
        '''
        print("\n\n\nStart processing ")
        
        # return if reaches the deepest level
        if level > self.level_cutoff:
            # add to pruned_clades
            self.pruned_clades.add(current_clade.name)
            print("At " + current_clade.name + ", the level limit exceeds.")
            return

        all_terminals = current_clade.get_terminals()
        if len(expr_cells) < self.min_cell_count_expr or len(all_terminals) < self.min_cell_count_cnv:
            self.pruned_clades.add(current_clade.name)
            if len(expr_cells) < self.min_cell_count_expr:
                print("At " + current_clade.name + ", there are less than " + str(self.min_cell_count_expr) + " cells in the expr matrix.")
            if len(all_terminals) < self.min_cell_count_cnv:
                print("At " + current_clade.name + ", there are less than " + str(self.min_cell_count_cnv) + " clades in the cnv matrix.")
            return

        # get next clades
        # given a clade, summarize diff cnv profile
        clades = current_clade.clades

        terminals = []
        clean_clades = []

        for cl in clades:
            current_terminals = [e.name for e in cl.get_terminals() if e.name in self.cnv_df.columns]
            if len(current_terminals) < self.min_cell_count_cnv:
                self.pruned_clades.add(cl.name)
            else:
                terminals.append(current_terminals)
                clean_clades.append(cl)

        # if there is only one clone left, add all scRNA cells to the clade
        if len(clean_clades) == 1:
            for cell in expr_cells:
                self.clone_assign_dict[cell] = clean_clades[0].name
            print("At " + current_clade.name + ", there is only one clean child clade existing.")
            self.assign_cells_to_clade(clean_clades[0], expr_cells, level + 1)
            return
            
        # if there is no clone, return
        if len(clean_clades) == 0:
            print("At " + current_clade.name + ", there is no clean child clade.")
            return
            
        # print the children
        for clean_clade in clean_clades:
            print("At " + current_clade.name + ", one of the child clade is " + clean_clade.name + " with " + str(len(current_terminals)) + " terminals. ")
            
            
        # construct total copy number input
        expr_input, clone_cnv_df = self.construct_total_copy_number_input(terminals, expr_cells)
        
        hscn_input, snv_allele_input, snv_input = self.construct_allele_specific_input(terminals, expr_cells)
        
        print("There are " + str(clone_cnv_df.shape[0]) + " genes in matrices. ")
        if clone_cnv_df.shape[0] < self.min_gene_diff and hscn_input.shape[0] < self.min_snp_diff:
            # add all clean clades to pruned clades
            for clade in clean_clades:
                self.pruned_clades.add(clade.name)
            print("cnv gene count less than self.min_gene_diff: " + str(clone_cnv_df.shape[0]))
            print("snp count less than self.min_snp_diff: "  + str(hscn_input.shape[0]))
            return

        # run clonealign
        print("Start run clonealign for clade: " + current_clade.name)
        print("cnv gene count: " + str(clone_cnv_df.shape[0]))
        print("expr cell count: " + str(expr_input.shape[1]))
        
        # record input
        if self.record_input_output:
            self.params_dict[current_clade.name] = dict()
            self.params_dict[current_clade.name]['input'] = dict()
            self.params_dict[current_clade.name]['input']['cnv'] = clone_cnv_df
            self.params_dict[current_clade.name]['input']['expr'] = expr_input
            self.params_dict[current_clade.name]['input']['hscn'] = hscn_input
            self.params_dict[current_clade.name]['input']['snv_allele'] = snv_allele_input
            self.params_dict[current_clade.name]['input']['snv'] = snv_input
        
        none_freq, clone_assign, clone_assign_df, params_dict = self.run_clonealign_pyro_repeat(clone_cnv_df, expr_input, hscn_input, snv_allele_input, snv_input)
        
        if self.record_input_output:
            self.params_dict[current_clade.name]['output'] = dict()
            self.params_dict[current_clade.name]['output']['none_freq'] = none_freq
            self.params_dict[current_clade.name]['output']['clone_assign'] = clone_assign
            self.params_dict[current_clade.name]['output']['clone_assign_df'] = clone_assign_df
            self.params_dict[current_clade.name]['output']['params_dict'] = params_dict
            

        print("Clonealign finished!")

        if 1 - none_freq < self.min_proceed_freq:
            print("CloneAlign Tree stops at clade: " + current_clade.name + " with correct frequency " + str(1 - none_freq) + '\n')
            for cl in clean_clades:
                self.pruned_clades.add(cl.name)
            return
        else:
            # record clone_assign
            print("CloneAlign Tree finishes at clade: " + current_clade.name + " with correct frequency " + str(1 - none_freq) + '\n')
            
            self.record_clone_assign_to_dict(expr_cells, clone_assign, clean_clades)
            if self.infer_s_score:
                self.record_param_to_dict(self.gene_type_score_dict, clone_cnv_df.index, params_dict['mean_gene_type_score'])
            
            if self.infer_b_allele:
                self.record_param_to_dict(self.allele_assign_prob_dict, hscn_df.index, params_dict['mean_allele_assign_prob'])
                

            for i in range(len(clean_clades)):
                new_expr_cells = [expr_cells[k] for k in range(len(expr_cells)) if clone_assign[k] == i]
                self.assign_cells_to_clade(clean_clades[i], new_expr_cells, level + 1)
            return
