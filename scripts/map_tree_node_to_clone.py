from Bio import Phylo
import pandas as pd
import argparse
import os

class ConvertNodeToClone:
    def __init__(self, tree, node_assign, clone_assign):
        self.tree = tree
        self.node_assign = node_assign
        self.clone_assign = clone_assign
        self.tree.ladderize()
        self.count = 0
        # add name for nodes if the nodes don't have name
        self.add_tree_node_name(self.tree.clade)

        self.nodes = self.node_assign["clone_id"].unique()
        self.node_clone_map = dict()

        self.map_node_to_clone(self.tree.clade)

        self.node_assign['clone_id'].replace(self.node_clone_map, inplace=True)


    def add_tree_node_name(self, node):
        if node.is_terminal():
            return
        if node.name is None:
            node.name = "node_" + str(self.count)
            self.count += 1
        for child in node.clades:
            self.add_tree_node_name(child)
        return

    def map_node_to_clone(self, current_clade, cut_off=0.9):
        if current_clade.is_terminal():
            return
        if current_clade.name in self.nodes:
            current_terminals = [terminal.name for terminal in current_clade.get_terminals() if terminal.name in self.clone_assign.index]
            clone_assign = self.clone_assign.loc[current_terminals, "clone_id"].value_counts()
            freq = clone_assign[0]/len(current_terminals)
            if freq >= cut_off:
                self.node_clone_map[current_clade.name] = clone_assign.index[0]
            else:
                self.node_clone_map[current_clade.name] = ""
        for child in current_clade.clades:
            self.map_node_to_clone(child, cut_off=cut_off)




def main():
    parser = argparse.ArgumentParser(description='map tree node to clone assignment')
    parser.add_argument('-t', '--tree', nargs=1, help='tree path')
    parser.add_argument('-n', '--node', nargs=1, help='node path')
    parser.add_argument('-c', '--clone', nargs=1, help='clone path')
    parser.add_argument('-o', '--output', nargs=1, help='output path')


    args = parser.parse_args()
    tree_path = args.tree[0]
    node_path = args.node[0]
    clone_path = args.clone[0]
    output_path = args.output[0]

    tree = Phylo.read(tree_path, "newick")
    node_assign = pd.read_csv(node_path, index_col=0)
    clone_assign = pd.read_csv(clone_path, index_col=0)

    convert = ConvertNodeToClone(tree, node_assign, clone_assign)
    convert.node_assign.to_csv(output_path, index=False)
    
    
if __name__ == "__main__":
    main()

