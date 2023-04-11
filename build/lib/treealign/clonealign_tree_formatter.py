"""
CloneAlignVis class
"""
from Bio import Phylo
import pandas as pd
import numpy as np


class TreeFormatter:

    @staticmethod
    def update_tree_node_assign(update_assign_dict, root, assigned_nodes, node_dict):
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
                TreeFormatter.update_tree_node_assign(update_assign_dict, clade, assigned_nodes, node_dict)
        elif count == 1:
            update_assign_dict[match_node] = root.name
        return    

    @staticmethod
    def get_all_non_terminal_nodes(clade, node_child_dict):
        if clade.is_terminal():
            return [clade.name]
        else:
            output = [i.name for i in clade.clades]
            for child in clade.clades:
                output = output + TreeFormatter.get_all_non_terminal_nodes(child, node_child_dict)
            node_child_dict[clade.name] = output
        return node_child_dict[clade.name]
    
    @staticmethod
    def clone_assign_df_to_dict(clone_assign_df):
        clone_assign_dict = {}
        for i in range(clone_assign_df.shape[0]):
            if clone_assign_df["clonealign_tree_id"].values[i] not in clone_assign_dict:
                clone_assign_dict[clone_assign_df["clonealign_tree_id"].values[i]] = []
            clone_assign_dict[clone_assign_df["clonealign_tree_id"].values[i]].append(
                clone_assign_df["cell_id"].values[i])
        return clone_assign_dict    
    

    @staticmethod
    def count_cells_in_clade(clade, clone_assign_dict, clade_cell_count_dict):
        count = 0
        if clade.is_terminal():
            clade_cell_count_dict[clade.name] = 0
            return 0
        else:
            if clade.name in clone_assign_dict:
                count += len(clone_assign_dict[clade.name])
            for child in clade.clades:
                count += TreeFormatter.count_cells_in_clade(child, clone_assign_dict, clade_cell_count_dict)
            clade_cell_count_dict[clade.name] = count
            return count
            
    @staticmethod
    def summarize_node_cell_freq(clade, clone_assign_dict, clade_cell_count_dict, result_dict):
        if clade.is_terminal():
            return
        else:
            current_result_dict = {}
            child_sum = 0
            for child in clade.clades:
                if clade_cell_count_dict[child.name] > 0:
                    current_result_dict[child.name] = clade_cell_count_dict[child.name]
                    child_sum += clade_cell_count_dict[child.name]
                    TreeFormatter.summarize_node_cell_freq(child, clone_assign_dict, clade_cell_count_dict, result_dict)

            remain_sum = clade_cell_count_dict[clade.name] - child_sum
            if len(current_result_dict) >= 2:
                current_result_dict[clade.name] = remain_sum
                result_dict[clade.name] = current_result_dict
            elif clade.name in clone_assign_dict:
                current_result_dict[clade.name] = remain_sum
                result_dict[clade.name] = current_result_dict
            return

    # find nodes that are inter-changable
    @staticmethod
    def find_interchangable_clades(clade, clade_cell_count_dict, interchangable_clades_dict):
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
    @staticmethod
    def generate_unify_node_name_dict(interchangable_clades_dict):
        roots = interchangable_clades_dict.keys()
        result_dict = {}
        for clade in roots:
            result_dict[clade] = TreeFormatter.unify_node_name(clade, interchangable_clades_dict)
        return result_dict
    
    # unify node name by referring to interchangable_clades_dict
    @staticmethod
    def unify_node_name(clade_name, interchangable_clades_dict):
        while clade_name in interchangable_clades_dict:
            clade_name = interchangable_clades_dict[clade_name]
        return clade_name

    @staticmethod
    def clean_tree_based_clonealign_output(tree, clone_assign):
        clade = tree.clade

        assigned_nodes = set(clone_assign["clonealign_tree_id"].values)

        update_assign_dict = {}
        node_child_dict = {}

        TreeFormatter.get_all_non_terminal_nodes(clade, node_child_dict)

        TreeFormatter.update_tree_node_assign(update_assign_dict, clade, assigned_nodes, node_child_dict)

        clone_assign = clone_assign.replace(update_assign_dict)

        clone_assign_dict = TreeFormatter.clone_assign_df_to_dict(clone_assign)

        # summarize number of cells assigned to each clade
        clade_cell_count_dict = {}
        result_dict = {}

        TreeFormatter.count_cells_in_clade(clade, clone_assign_dict, clade_cell_count_dict)

        TreeFormatter.summarize_node_cell_freq(clade, clone_assign_dict, clade_cell_count_dict, result_dict)

        interchangable_clades_dict = {}
        TreeFormatter.find_interchangable_clades(clade, clade_cell_count_dict, interchangable_clades_dict)

        unify_dict = TreeFormatter.generate_unify_node_name_dict(interchangable_clades_dict)
        clone_assign = clone_assign.replace(unify_dict)

        result_list = []
        for node in result_dict:
            current_result = {"name": TreeFormatter.unify_node_name(node, interchangable_clades_dict), "value": []}
            for child_node in result_dict[node]:
                current_result["value"].append({"name": TreeFormatter.unify_node_name(child_node, interchangable_clades_dict),
                                                "value": result_dict[node][child_node]})
            result_list.append(current_result)

        return clone_assign, result_list

    @staticmethod
    def get_cnv_cell_assignments(clone_assign, tree, cnv_cells):
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
