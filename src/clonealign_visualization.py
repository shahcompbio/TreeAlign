"""
CloneAlignVis class
"""
from Bio import Phylo
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import simplejson as json


class CloneAlignVis:
    CHR_DICT = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15,
                '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '22': 22, 'X': 23, 'Y': 24}

    def __init__(self, genes, tree, cnv_matrix=None, expr_matrix=None,
                 clone_assign_clone=None, clone_assign_tree=None, cnv_meta=None, expr_meta=None,
                 total_gene_count=2000, generate_sankey=True,
                 expr_cell_order=['clonealign_tree_id', 'clonealign_clone_id', 'infercnv_cluster_id', 'sample_id']):
        self.sankey = []
        self.genes = genes
        self.expr_cell_order = expr_cell_order

        self.tree = tree
        self.tree.ladderize()
        self.count = 0
        # add name for nodes if the nodes don't have name
        self.add_tree_node_name(self.tree.clade)

        self.cnv_matrix = cnv_matrix
        self.expr_matrix = expr_matrix
        # rename column names
        self.clone_assign_clone = clone_assign_clone
        self.clone_assign_clone = self.clone_assign_clone.rename(columns={'clone_id': 'clonealign_clone_id'})
        self.clone_assign_tree = clone_assign_tree
        self.clone_assign_tree = self.clone_assign_tree.rename(columns={'clone_id': 'clonealign_tree_id'})

        self.expr_meta = expr_meta
        self.cnv_meta = cnv_meta
        self.cnv_meta = self.cnv_meta.rename(columns={'clone_id': 'clonealign_clone_id'})

        self.total_gene_count = total_gene_count

        # if tree is not None, get cnv cell order df from tree. generate consensus data accordingly

        self.cnv_cells = pd.DataFrame({'cell_id': [terminal.name for terminal in tree.get_terminals()]})
        # if we have both tree and tree-based clonealign results
        if self.clone_assign_tree is not None:
            # get clean clone assign tree results
            self.clone_assign_tree, self.pie_chart = self.clean_tree_based_clonealign_output(self.tree,
                                                                                             self.clone_assign_tree)
            # get terminal nodes
            self.terminal_nodes = []
            for entry in self.pie_chart:
                if len(entry["value"]) == 1:
                    self.terminal_nodes.append(entry["name"])

            # get cnv cell assignments
            self.cnv_clone_assign = self.get_cnv_cell_assignments()
        else:
            self.clone_assign_tree = None
            self.pie_chart = None
            self.cnv_clone_assign = None

        # merge all cnv meta data
        self.cnv_meta = self.merge_meta(self.cnv_cells, 'left', self.cnv_meta, self.cnv_clone_assign)

        # clean up all the expr meta data
        self.expr_cells = pd.DataFrame({'cell_id': self.expr_matrix.columns.values.tolist()})

        # else order cnv cells by clone_id
        self.expr_meta = self.merge_meta(self.expr_cells, 'inner', self.expr_meta, self.clone_assign_tree,
                                         self.clone_assign_clone)

        # replace nan with empty string
        self.cnv_meta = self.cnv_meta.replace(np.nan, "", regex=True)
        self.expr_meta = self.expr_meta.replace(np.nan, "", regex=True)

        # re-order cells by EXPR_CELL_ORDER
        self.order_expr_cells(generate_sankey)
        self.expr_cells = pd.DataFrame({'cell_id': self.expr_meta['cell_id'].values.tolist()})

        # get consensus genes
        self.genes = self.get_consensus_genes()

        self.cnv_matrix = self.cnv_matrix.reindex(self.genes['gene'].values.tolist())
        self.expr_matrix = self.expr_matrix.reindex(self.genes['gene'].values.tolist())

        self.cnv_matrix = self.cnv_matrix.reindex(columns=self.cnv_meta['cell_id'].values.tolist())
        self.expr_matrix = self.expr_matrix.reindex(columns=self.expr_meta['cell_id'].values.tolist())

        # subsample the matrix to keep given number of genes
        self.subsample_genes()

        # bin float expr to discrete
        self.bin_expr_matrix()

    def output_json(self):
        output = dict()
        if self.tree is not None:
            root = self.tree.clade

            def get_json(clade):
                js_output = {"name": clade.name}
                if not clade.is_terminal():
                    clades = clade.clades
                    js_output["children"] = []
                    for clade in clades:
                        js_output["children"].append(get_json(clade))
                return js_output

            json_dict = get_json(root)
            output['tree'] = json_dict

        if self.pie_chart is not None:
            output['pie_chart'] = self.pie_chart

        if self.expr_meta is not None:
            output['expr_meta'] = self.expr_meta.to_dict('list')

        if self.cnv_meta is not None:
            output['cnv_meta'] = self.cnv_meta.to_dict('list')

        if self.expr_matrix is not None:
            output['expr_matrix'] = self.convert_cell_gene_matrix_to_list(self.expr_matrix)

        if self.cnv_matrix is not None:
            output['cnv_matrix'] = self.convert_cell_gene_matrix_to_list(self.cnv_matrix)

        if self.sankey is not None and len(self.sankey) > 0:
            output['sankey'] = self.sankey

        if self.terminal_nodes is not None:
            output['terminal_nodes'] = self.terminal_nodes
        return output

    @staticmethod
    def pack_into_tab_data(output_json_file, data, tab_titles=None, tab_contents=None):
        def convert(o):
            if isinstance(o, np.int64):
                return int(o)
            raise TypeError

        output = []
        for i in range(len(data)):
            tab_data = {'id': str(i), 'tabTitle': tab_titles[i], 'tabContent': tab_contents[i], 'data': data[i]}
            output.append(tab_data)
        with open(output_json_file, 'w') as f:
            output_json = json.dumps(output, separators=(',', ':'), sort_keys=False, ignore_nan=True, default=convert)
            f.write(output_json)
        return

    def add_tree_node_name(self, node):
        if node.is_terminal():
            return
        if node.name is None:
            node.name = "node_" + str(self.count)
            self.count += 1
        for child in node.clades:
            self.add_tree_node_name(child)
        return

    def bin_expr_matrix(self, n_bins=15):
        expr_array = self.expr_matrix.values.flatten()
        # construct bins
        bin_width = (np.median(expr_array) - expr_array.min()) / int(n_bins / 2)
        min_value = np.median(expr_array) - bin_width * n_bins / 2
        bins = [min_value]
        for i in range(n_bins):
            bins.append(min_value + (i + 1) * bin_width)

        bins[len(bins) - 1] = expr_array.max()
        self.expr_matrix = self.expr_matrix.apply(pd.cut, bins=bins, labels=range(n_bins))
        return

    def convert_cell_gene_matrix_to_list(self, matrix):
        matrix = matrix.astype('int32')
        matrix = matrix.transpose()
        output_list = []
        for i in list(self.genes['chr'].unique()):
            chr_dict = {'chr': i}
            chr_matrix = matrix.loc[:, (self.genes["chr"] == i).values]
            chr_matrix = chr_matrix.to_numpy()
            chr_matrix_array = []
            for array in chr_matrix:
                array_list = [number.item() for number in array]
                chr_matrix_array.append(array_list)
            chr_dict['value'] = chr_matrix_array
            output_list.append(chr_dict)
        return output_list

    def merge_meta(self, cell_order, how, *args):
        output = cell_order
        for arg in args:
            if arg is not None:
                output = output.merge(arg, how=how, on='cell_id')
        return output

    def subsample_genes(self):
        gene_group = int(self.genes.shape[0] / self.total_gene_count)
        select_rows = [i for i in range(self.genes.shape[0]) if i % gene_group == 1]
        self.genes = self.genes.iloc[select_rows]
        self.cnv_matrix = self.cnv_matrix.iloc[select_rows]
        self.expr_matrix = self.expr_matrix.iloc[select_rows]

    def order_chromosome(self, input_chr_series):
        if is_numeric_dtype(input_chr_series):
            return input_chr_series
        else:
            return input_chr_series.replace(self.CHR_DICT)

    def get_consensus_genes(self):
        genes_list = []
        if self.cnv_matrix is not None:
            cnv_genes = pd.DataFrame({'gene': self.cnv_matrix.index.values.tolist()})
            genes_list.append(cnv_genes)
        if self.expr_matrix is not None:
            expr_genes = pd.DataFrame({'gene': self.expr_matrix.index.values.tolist()})
            genes_list.append(expr_genes)
        output = self.genes
        for i in range(len(genes_list)):
            output = output.merge(genes_list[i], on='gene')
        # order genes by chromosome locations
        output = output.sort_values(by=['chr', 'start'], key=self.order_chromosome, ignore_index=True)
        return output

    def order_expr_cells(self, generateSankey=True):
        order_columns = [order_column for order_column in self.expr_cell_order if
                         order_column in self.expr_meta.columns.values]
        # if the first column is also present in self.cnv_meta, match up with self.cnv_meta
        categories = [i for i in self.cnv_meta[order_columns[0]].unique().tolist() if i is not None]
        for category in self.expr_meta[order_columns[0]].unique().tolist():
            if category is not None and category not in categories:
                categories.append(category)
        if generateSankey and order_columns[0] in self.cnv_meta.columns.values:
            self.cnv_meta[order_columns[0]] = pd.Categorical(self.cnv_meta[order_columns[0]], categories, ordered=True)
            self.expr_meta[order_columns[0]] = pd.Categorical(self.expr_meta[order_columns[0]], categories,
                                                              ordered=True)
        self.expr_meta = self.expr_meta.sort_values(by=order_columns, ignore_index=True)

        if generateSankey:
            self.generate_sankey(order_columns[0])
        return

    def generate_sankey(self, select_column):
        for terminal in self.terminal_nodes:
            left_indices = self.cnv_meta.index[self.cnv_meta[select_column] == terminal].values
            right_indices = self.expr_meta.index[self.expr_meta[select_column] == terminal].values
            sankey_element = {"name": terminal,
                              "left": [left_indices.min().item(), left_indices.max().item()],
                              "right": [right_indices.min().item(), right_indices.max().item()]}
            self.sankey.append(sankey_element)

    def get_cnv_cell_assignments(self):
        clone_assign = self.clone_assign_tree
        tree = self.tree
        cnv_cells = self.cnv_cells

        clones = set(clone_assign['clonealign_tree_id'].unique().tolist())
        cnv_clone_assign = pd.DataFrame(
            {'cell_id': cnv_cells['cell_id'].values, 'clonealign_tree_id': [None] * cnv_cells.shape[0]})

        clade = tree.clade
        node_queue = [clade]
        while len(node_queue) > 0:
            current_len = len(node_queue)
            for i in range(current_len):
                current_clade = node_queue.pop(0)
                if current_clade.name in clones:
                    current_terminals = [terminal.name for terminal in current_clade.get_terminals()]
                    cnv_clone_assign.loc[
                        cnv_clone_assign['cell_id'].isin(current_terminals), 'clonealign_tree_id'] = current_clade.name
                if not current_clade.is_terminal():
                    for child in current_clade.clades:
                        node_queue.append(child)

        return cnv_clone_assign

    def get_all_non_terminal_nodes(self, clade, node_child_dict):
        if clade.is_terminal():
            return [clade.name]
        else:
            output = [i.name for i in clade.clades]
            for child in clade.clades:
                output = output + self.get_all_non_terminal_nodes(child, node_child_dict)
            node_child_dict[clade.name] = output;
        return node_child_dict[clade.name]

    def update_tree_node_assign(self, update_assign_dict, root, assigned_nodes, node_dict):
        if root.is_terminal():
            return
        all_nodes = node_dict[root.name]
        count = 0
        for node in all_nodes:
            if node in assigned_nodes:
                count += 1
                match_node = node
        if count >= 2:
            for clade in root.clades:
                self.update_tree_node_assign(update_assign_dict, clade, assigned_nodes, node_dict)
        elif count == 1:
            update_assign_dict[match_node] = root.name
        return

    def clone_assign_df_to_dict(self, clone_assign_df):
        clone_assign_dict = {}
        for i in range(clone_assign_df.shape[0]):
            if clone_assign_df["clonealign_tree_id"].values[i] not in clone_assign_dict:
                clone_assign_dict[clone_assign_df["clonealign_tree_id"].values[i]] = []
            clone_assign_dict[clone_assign_df["clonealign_tree_id"].values[i]].append(
                clone_assign_df["cell_id"].values[i])
        return clone_assign_dict

    def count_cells_in_clade(self, clade, clone_assign_dict, clade_cell_count_dict):
        count = 0
        if clade.is_terminal():
            clade_cell_count_dict[clade.name] = 0
            return 0
        else:
            if clade.name in clone_assign_dict:
                count += len(clone_assign_dict[clade.name])
            for child in clade.clades:
                count += self.count_cells_in_clade(child, clone_assign_dict, clade_cell_count_dict)
            clade_cell_count_dict[clade.name] = count
            return count

    def summarize_node_cell_freq(self, clade, clone_assign_dict, clade_cell_count_dict, result_dict):
        if clade.is_terminal():
            return
        else:
            current_result_dict = {}
            child_sum = 0
            for child in clade.clades:
                if clade_cell_count_dict[child.name] > 0:
                    current_result_dict[child.name] = clade_cell_count_dict[child.name]
                    child_sum += clade_cell_count_dict[child.name]
                    self.summarize_node_cell_freq(child, clone_assign_dict, clade_cell_count_dict, result_dict)

            remain_sum = clade_cell_count_dict[clade.name] - child_sum
            if len(current_result_dict) >= 2:
                current_result_dict[clade.name] = remain_sum
                result_dict[clade.name] = current_result_dict
            elif clade.name in clone_assign_dict:
                current_result_dict[clade.name] = remain_sum
                result_dict[clade.name] = current_result_dict
            return

    # find nodes that are inter-changable
    def find_interchangable_clades(self, clade, clade_cell_count_dict, interchangable_clades_dict):
        node_queue = []
        if not clade.is_terminal():
            node_queue.append(clade)
        while len(node_queue) > 0:
            current_len = len(node_queue)
            for i in range(current_len):
                current_node = node_queue.pop(0)
                for child in current_node.clades:
                    if clade_cell_count_dict[child.name] > 0:
                        if clade_cell_count_dict[child.name] == clade_cell_count_dict[current_node.name]:
                            interchangable_clades_dict[current_node.name] = child.name
                        if not child.is_terminal():
                            node_queue.append(child)

    # unify node name by referring to interchangable_clades_dict
    def unify_node_name(self, clade_name, interchangable_clades_dict):
        while clade_name in interchangable_clades_dict:
            clade_name = interchangable_clades_dict[clade_name]
        return clade_name

    def generate_unify_node_name_dict(self, interchangable_clades_dict):
        roots = interchangable_clades_dict.keys()
        result_dict = {}
        for clade in roots:
            result_dict[clade] = self.unify_node_name(clade, interchangable_clades_dict)
        return result_dict

        # generate cleaned tree-based clone assignment meta data and node pie chart annotations

    # return
    def clean_tree_based_clonealign_output(self, tree, clone_assign):
        clade = tree.clade

        assigned_nodes = set(clone_assign["clonealign_tree_id"].values)

        update_assign_dict = {}
        node_child_dict = {}

        self.get_all_non_terminal_nodes(clade, node_child_dict)

        self.update_tree_node_assign(update_assign_dict, clade, assigned_nodes, node_child_dict)

        clone_assign = clone_assign.replace(update_assign_dict)

        clone_assign_dict = self.clone_assign_df_to_dict(clone_assign)

        # summarize number of cells assigned to each clade
        clade_cell_count_dict = {}
        result_dict = {}

        self.count_cells_in_clade(clade, clone_assign_dict, clade_cell_count_dict)

        self.summarize_node_cell_freq(clade, clone_assign_dict, clade_cell_count_dict, result_dict)

        interchangable_clades_dict = {}
        self.find_interchangable_clades(clade, clade_cell_count_dict, interchangable_clades_dict)

        unify_dict = self.generate_unify_node_name_dict(interchangable_clades_dict)
        clone_assign = clone_assign.replace(unify_dict)

        result_list = []
        for node in result_dict:
            current_result = {"name": self.unify_node_name(node, interchangable_clades_dict), "value": []}
            for child_node in result_dict[node]:
                current_result["value"].append({"name": self.unify_node_name(child_node, interchangable_clades_dict),
                                                "value": result_dict[node][child_node]})
            result_list.append(current_result)

        return clone_assign, result_list
