from treealign import CloneAlignClone
from treealign import CloneAlignTree

import pandas as pd
from Bio import Phylo

expr = pd.read_csv("data/example_expr.csv", index_col=0)
cnv = pd.read_csv("data/example_gene_cnv.csv", index_col=0)
clone = pd.read_csv("data/example_cell_clone.csv")

tree = Phylo.read("data/example_phylogeny.newick", "newick")

#clonealignclone_obj = CloneAlignClone(expr, cnv, clone, repeat=1)
#clonealignclone_obj.assign_cells_to_clones()

obj = CloneAlignClone(expr, cnv, clone, repeat=2)
obj.assign_cells_to_clones()

clone_assign, gene_type_score, clone_assign_raw, gene_type_score_raw = obj.generate_output()

clone_assign.to_csv("data/clone_assign_test.csv")
gene_type_score.to_csv("data/gene_type_score_test.csv")
