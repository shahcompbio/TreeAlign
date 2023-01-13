import unittest
from pathlib import Path
from Bio import Phylo
import pandas as pd
import filecmp

from treealign import CloneAlignVis

class TestOutputJson(unittest.TestCase):
    DATA_DIR = Path(__file__).parent / 'data'
    def test_output_json_consistency(self):
        """
        Test json output for sample data holds
        """
        genes_format = self.DATA_DIR / 'geneAnnotation.txt'
        tree_format = self.DATA_DIR / 'SPECTRUM-OV-022_hdbscan_total.newick'
        cnv_matrix_format = self.DATA_DIR / 'SPECTRUM-OV-022_gene_cnv.csv'
        expr_matrix_format = self.DATA_DIR / 'SPECTRUM-OV-022_expr.infercnv.dat'
        clone_assign_clone_format = self.DATA_DIR / 'SPECTRUM-OV-022_clone_assignment_clone_total_cn.csv'
        clone_assign_tree_format = self.DATA_DIR / 'SPECTRUM-OV-022_clone_assignment_tree_integrated.csv'
        cnv_clone_format = self.DATA_DIR / 'SPECTRUM-OV-022_cell_clone_total.csv'
        cnv_meta_format = self.DATA_DIR / 'SPECTRUM-OV-022_cell_clone_total.csv'
        expr_meta_format = self.DATA_DIR / 'SPECTRUM-OV-022_meta.csv'
        infercnv_meta_format = self.DATA_DIR / 'SPECTRUM-OV-022_rand_tree_tumor_subcluters.csv'

        genes = pd.read_csv(genes_format, sep = "\t")
        genes.columns = ["gene", "chr", "start", "end"]

        tree = Phylo.read(tree_format, "newick")
        cnv_matrix = pd.read_csv(cnv_matrix_format, index_col = 0)
        expr_matrix = pd.read_csv(expr_matrix_format, sep = "\t", index_col = 0)
        clone_assign_clone = pd.read_csv(clone_assign_clone_format, index_col = 0)
        clone_assign_tree = pd.read_csv(clone_assign_tree_format, index_col = 0)
        cnv_clone = pd.read_csv(cnv_clone_format)
        cnv_meta = pd.read_csv(cnv_meta_format)
        cnv_meta = cnv_clone.merge(cnv_meta)
        expr_meta = pd.read_csv(expr_meta_format)  
        infercnv_meta = pd.read_csv(infercnv_meta_format)
        infercnv_meta.columns = ["infercnv_cluster_name", "cell_id"]
        expr_meta = expr_meta.merge(infercnv_meta)            

        clonealign_vis = CloneAlignVis(genes, tree, cnv_matrix, expr_matrix, clone_assign_clone, clone_assign_tree, cnv_meta, expr_meta, 1000)
        json_output = clonealign_vis.output_json()

        CloneAlignVis.pack_into_tab_data(self.DATA_DIR / "test.json", [json_output], tab_titles=["SPECTRUM-OV-022"], tab_contents=["SPECTRUM-OV-022"])

        self.assertTrue(filecmp.cmp(self.DATA_DIR / 'test.json', self.DATA_DIR / 'OV_022_integrated.json'))

if __name__ == '__main__':
    unittest.main()        