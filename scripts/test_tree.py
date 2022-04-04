from treealign import CloneAlignClone
from treealign import CloneAlignTree

import pandas as pd
from Bio import Phylo

expr = pd.read_csv("data/example_expr.csv", index_col=0)
cnv = pd.read_csv("data/example_gene_cnv.csv", index_col=0)

tree = Phylo.read("data/example_phylogeny.newick", "newick")

clonealignclone_obj = CloneAlignTree(expr, cnv, tree, repeat=1, min_gene_diff=50)
clonealignclone_obj.assign_cells_to_tree()


clone_assign, gene_type_score = clonealignclone_obj.generate_output()

clone_assign.to_csv("data/clone_assign_test.csv")
gene_type_score.to_csv("data/gene_type_score_test.csv")
