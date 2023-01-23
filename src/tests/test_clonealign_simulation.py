import unittest
from pathlib import Path
import pandas as pd
import filecmp
import random

from treealign.clonealign_simulation import CloneAlignSimulation

class TestCloneAlignSimulation(unittest.TestCase):
    DATA_DIR = Path(__file__).parent / 'data/simulation'

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)


    def test_clonealign_clone_total_cn_simulation(self):
        """
        Test simulation output of 
        """
        expr = pd.read_csv(self.DATA_DIR / "total_cn/SPECTRUM-OV-081_expr.csv", index_col=0)

        cnv = pd.read_csv(self.DATA_DIR / "total_cn/SPECTRUM-OV-081_gene_cnv.csv", index_col=0)

        clone = pd.read_csv(self.DATA_DIR / "total_cn/SPECTRUM-OV-081_cell_clone.csv")

        random.seed(12)
        obj = CloneAlignSimulation(expr, cnv, clone)
        obj.simulate_data(str(self.DATA_DIR / "test/total_cn"), index=0, gene_count=100, cell_counts=[100], cnv_dependency_freqs=[0.1])

        self.assertTrue(filecmp.cmp(self.DATA_DIR / "total_cn/simulated_clone_assignment_gene_100_snp_500_cell_100_cnv_0.1_index_0.csv", self.DATA_DIR / "test/total_cn/simulated_clone_assignment_gene_100_snp_500_cell_100_cnv_0.1_index_0.csv"))

        self.assertTrue(filecmp.cmp(self.DATA_DIR / "total_cn/simulated_expr_gene_100_snp_500_cell_100_cnv_0.1_index_0.csv", self.DATA_DIR / "test/total_cn/simulated_expr_gene_100_snp_500_cell_100_cnv_0.1_index_0.csv"))

        self.assertTrue(filecmp.cmp(self.DATA_DIR / "total_cn/simulated_gene_type_score_gene_100_snp_500_cell_100_cnv_0.1_index_0.csv", self.DATA_DIR / "test/total_cn/simulated_gene_type_score_gene_100_snp_500_cell_100_cnv_0.1_index_0.csv"))

    def test_clonealign_clone_allele_simulation(self):
        expr = pd.read_csv(self.DATA_DIR / "allele_specific/SPECTRUM-OV-118_expr.csv", index_col=0)

        cnv = pd.read_csv(self.DATA_DIR / "allele_specific/SPECTRUM-OV-118_gene_cnv.csv", index_col=0)

        clone = pd.read_csv(self.DATA_DIR / "allele_specific/SPECTRUM-OV-118_cell_clone_total.csv")

        hscn = pd.read_csv(self.DATA_DIR / "allele_specific/SPECTRUM-OV-118_snp_baf.csv", index_col=0)     

        snv_allele = pd.read_csv(self.DATA_DIR / "allele_specific/SPECTRUM-OV-118_snp_allele.csv", index_col=0)

        snv = pd.read_csv(self.DATA_DIR / "allele_specific/SPECTRUM-OV-118_snp_total.csv", index_col=0)        

        random.seed(12)
        obj = CloneAlignSimulation(expr, cnv, clone, hscn, snv_allele, snv)
        obj.simulate_data(str(self.DATA_DIR / "test/allele_specific"), index=0, gene_count=100, snp_count=100, cell_counts=[100], cnv_dependency_freqs=[0.1])

        self.assertTrue(filecmp.cmp(self.DATA_DIR / "allele_specific/simulated_clone_assignment_gene_100_snp_100_cell_100_cnv_0.1_index_0.csv", self.DATA_DIR / "test/allele_specific/simulated_clone_assignment_gene_100_snp_100_cell_100_cnv_0.1_index_0.csv"))

        self.assertTrue(filecmp.cmp(self.DATA_DIR / "allele_specific/simulated_expr_gene_100_snp_100_cell_100_cnv_0.1_index_0.csv", self.DATA_DIR / "test/allele_specific/simulated_expr_gene_100_snp_100_cell_100_cnv_0.1_index_0.csv"))

        self.assertTrue(filecmp.cmp(self.DATA_DIR / "allele_specific/simulated_gene_type_score_gene_100_snp_100_cell_100_cnv_0.1_index_0.csv", self.DATA_DIR / "test/allele_specific/simulated_gene_type_score_gene_100_snp_100_cell_100_cnv_0.1_index_0.csv"))


        self.assertTrue(filecmp.cmp(self.DATA_DIR / "allele_specific/simulated_snv_allele_gene_100_snp_100_cell_100_cnv_0.1_index_0.csv", self.DATA_DIR / "test/allele_specific/simulated_snv_allele_gene_100_snp_100_cell_100_cnv_0.1_index_0.csv"))

        self.assertTrue(filecmp.cmp(self.DATA_DIR / "allele_specific/simulated_snv_gene_100_snp_100_cell_100_cnv_0.1_index_0.csv", self.DATA_DIR / "test/allele_specific/simulated_snv_gene_100_snp_100_cell_100_cnv_0.1_index_0.csv"))    

if __name__ == '__main__':
    unittest.main()        