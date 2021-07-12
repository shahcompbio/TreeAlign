from src.clonealign_clone import CloneAlignClone
from src.clonealign_tree import CloneAlignTree

import pandas as pd
from Bio import Phylo

expr = pd.read_csv("/Users/shih/pyro_tutorial/clonealign_pyro/data/SA1054_expr.csv", index_col=0)
cnv = pd.read_csv("/Users/shih/pyro_tutorial/clonealign_pyro/data/SA1054_gene_cnv.csv", index_col=0)
clone = pd.read_csv("/Users/shih/pyro_tutorial/clonealign_pyro/data/SA1054_cnv_meta.csv")

tree = Phylo.read("/Users/shih/pyro_tutorial/clonealign_pyro/data/SA1054_tree.newick", "newick")

#clonealignclone_obj = CloneAlignClone(expr, cnv, clone, repeat=1)
#clonealignclone_obj.assign_cells_to_clones()

obj = CloneAlignClone(expr, cnv, clone, repeat=2)
obj.assign_cells_to_clones()

clone_assign, gene_type_score, clone_assign_raw, gene_type_score_raw = obj.generate_output()

clone_assign.to_csv("clone_assign_test.csv")
gene_type_score.to_csv("gene_type_score_test.csv")
clone_assign_raw.to_csv("clone_assign_test_raw.csv")
gene_type_score_raw.to_csv("gene_type_score_test_raw.csv")