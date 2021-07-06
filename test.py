from src.clonealign_clone import CloneAlignClone
from src.clonealign_tree import CloneAlignTree

import pandas as pd
from Bio import Phylo

expr = pd.read_csv("/Users/shih/pyro_tutorial/clonealign_pyro/data/SPECTRUM-OV-022_expr_clonealign_input.csv", index_col=0)
cnv = pd.read_csv("/Users/shih/pyro_tutorial/clonealign_pyro/SPECTRUM-OV-022_gene_cnv.csv", index_col=0)
clone = pd.read_csv("/Users/shih/pyro_tutorial/clonealign_pyro/SPECTRUM-OV-022_cell_clones.csv")

tree = Phylo.read("/Users/shih/pyro_tutorial/clonealign_pyro/tree0.newick", "newick")

#clonealignclone_obj = CloneAlignClone(expr, cnv, clone, repeat=1)
#clonealignclone_obj.assign_cells_to_clones()

clonealigntree_obj = CloneAlignTree(expr, cnv, tree, repeat=1)
clonealigntree_obj.assign_cells_to_tree()

clone_assign, gene_type_score = clonealigntree_obj.generate_output()

clone_assign.to_csv("clone_assign_test.csv")
gene_type_score.to_csv("gene_type_score_test.csv")