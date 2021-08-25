"""
CloneAlignSimulation class
"""
import torch
import os
import pyro
import pyro.distributions as dist
from torch.nn import Softplus
import pandas as pd
import random
from .clonealign_clone import CloneAlignClone


class CloneAlignSimulation:
    def __init__(self, expr, cnv, clone):
        # run CloneAlignClone on real data
        obj = CloneAlignClone(expr, cnv, clone, normalize_cnv=True, cnv_cutoff=10, model_select="gene", repeat=1,
                              min_clone_cell_count=20,
                              min_clone_assign_prob=0.8, min_clone_assign_freq=0.7, min_consensus_gene_freq=0.6,
                              max_temp=1.0, min_temp=0.5, anneal_rate=0.01, learning_rate=0.1, max_iter=400,
                              rel_tol=5e-5)
        summarized_clone_assign, summarized_gene_type_score, clone_assign_df, gene_type_score_df = obj.assign_cells_to_clones()
        self.summarized_gene_type_score = summarized_gene_type_score
        self.map_estimates = obj.map_estimates
        self.cnv = obj.cnv_df
        self.clone = obj.clone_df

    def simulate_data(self, output_dir, gene_count=500, cell_counts=[100, 500, 1000],
                      cnv_dependency_freqs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):

        cell_counts.sort()
        cnv_dependency_freqs.sort()

        cell_samples = []

        cell_samples.append(random.choices(range(self.cnv.shape[1]), k=cell_counts[0]))
        for i in range(1, len(cell_counts)):
            current_samples = random.choices(range(self.cnv.shape[1]), k=cell_counts[i] - cell_counts[i - 1])
            cell_samples.append(cell_samples[i - 1] + current_samples)

        gene_ids = random.sample(range(self.summarized_gene_type_score.shape[0]), k=gene_count)

        gene_type_score_dict = {}
        for cnv_dependency_freq in cnv_dependency_freqs:
            gene_type_score_dict[cnv_dependency_freq] = [1] * int(gene_count * cnv_dependency_freq) + [0] * (
                    gene_count - int(gene_count * cnv_dependency_freq))

        gene_type_score_df = pd.DataFrame(gene_type_score_dict)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        simulated_clone_assignment_format = output_dir + "/simulated_clone_assignment_cell_{cell_count}_cnv_{cnv_freq}.csv"
        simulated_gene_type_score_format = output_dir + "/simulated_gene_type_score_cell_{cell_count}_cnv_{cnv_freq}.csv"
        simulated_expr_format = output_dir + "/simulated_expr_cell_{cell_count}_cnv_{cnv_freq}.csv"

        for i in range(len(cnv_dependency_freqs)):
            for j in range(len(cell_samples)):
                current_freq = cnv_dependency_freqs[i]
                cell_sample = cell_samples[j]

                simulated_cell_assignment, gene_type_score_simulated, expected_expr_df = self.simulate_individual_data(
                    gene_ids, cell_sample, gene_type_score_df[current_freq])
                simulated_cell_assignment.to_csv(
                    simulated_clone_assignment_format.format(cell_count=len(cell_sample), cnv_freq=current_freq))
                gene_type_score_simulated.to_csv(
                    simulated_gene_type_score_format.format(cell_count=len(cell_sample), cnv_freq=current_freq))
                expected_expr_df.to_csv(
                    simulated_expr_format.format(cell_count=len(cell_sample), cnv_freq=current_freq))

    def simulate_individual_data(self, gene_ids, cell_sample, gene_type_score):

        current_cnv = self.cnv.iloc[gene_ids, cell_sample]
        current_cnv = torch.tensor(current_cnv.values).transpose(0, 1)

        per_copy_expr = self.map_estimates['expose_per_copy_expr'][gene_ids]
        w = self.map_estimates['expose_w'][gene_ids]
        psi = self.map_estimates['expose_psi'][cell_sample]

        softplus = Softplus()

        # calculate copy number mean
        per_copy_expr = softplus(per_copy_expr)

        gene_type_score = torch.tensor(gene_type_score.values)

        with pyro.plate('cell', len(cell_sample)):
            expected_expr = (per_copy_expr * current_cnv * gene_type_score +
                             per_copy_expr * (1 - gene_type_score)) * \
                            torch.exp(torch.matmul(psi, torch.transpose(w, 0, 1)))

            # draw expr from Multinomial
            expr_simulated = pyro.sample('obs',
                                         dist.Multinomial(total_count=3000, probs=expected_expr, validate_args=False))


        expected_expr_df = pd.DataFrame(expr_simulated.transpose(0, 1).detach().numpy())
        expected_expr_df.index = self.summarized_gene_type_score['gene'][gene_ids]

        gene_type_score_simulated = pd.DataFrame(
            {'gene': expected_expr_df.index.values, 'gene_type_score': gene_type_score.detach().numpy()})

        cell_sample_names = self.cnv.columns.values[cell_sample].tolist()
        simulated_cell_assignment = pd.DataFrame(
            {'cell_id': cell_sample_names, 'expr_cell_id': expected_expr_df.columns.values.tolist()})

        simulated_cell_assignment = simulated_cell_assignment.merge(self.clone, how='left')

        return simulated_cell_assignment, gene_type_score_simulated, expected_expr_df
