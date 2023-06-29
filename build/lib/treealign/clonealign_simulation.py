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
from pyro.ops.indexing import Vindex
from .clonealign_clone import CloneAlignClone
from torch.distributions.binomial import Binomial


class CloneAlignSimulation:
    def __init__(self, expr, cnv, clone, hscn=None, snv_allele=None, snv=None):
        # run CloneAlignClone on real data
        obj = CloneAlignClone(clone, expr, cnv, hscn, snv_allele, snv, repeat=1, record_input_output=True, infer_b_allele=False, normalize_cnv=True, min_consensus_snv_freq=0.3)
        obj.assign_cells_to_clones()
        self.map_estimates = obj.map_estimates
        self.clone_cnv_df = obj.params_dict['input']['cnv']
        self.clone = obj.clone_df

        self.hscn = obj.params_dict['input']['hscn']
        self.snv_allele = obj.params_dict['input']['snv_allele']
        self.snv = obj.params_dict['input']['snv']

    def simulate_data(self, output_dir, index=1, gene_count=500, snp_count=500, cell_counts=[100, 1000, 5000],
                      cnv_dependency_freqs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], add_snp_read_count_prob=0.1, add_noises_per_cell=2000):
        if self.hscn is not None and self.snv_allele is not None and self.snv is not None and snp_count > 0:
            self.generate_allelic_data = True
        else:
            self.generate_allelic_data = False

        self.index = index
        cell_counts.sort()
        cnv_dependency_freqs.sort()

        cell_samples = []

        # sample cells
        cell_samples.append(random.choices(range(self.clone_cnv_df.shape[1]), k=cell_counts[0]))
        for i in range(1, len(cell_counts)):
            current_samples = random.choices(range(self.clone_cnv_df.shape[1]), k=cell_counts[i] - cell_counts[i - 1])
            cell_samples.append(cell_samples[i - 1] + current_samples)

        # sample genes
        gene_ids = random.sample(range(self.clone_cnv_df.shape[0]), k=gene_count)

        gene_type_score_dict = {}
        for cnv_dependency_freq in cnv_dependency_freqs:
            gene_type_score_dict[cnv_dependency_freq] = [1] * int(gene_count * cnv_dependency_freq) + [0] * (
                    gene_count - int(gene_count * cnv_dependency_freq))

        gene_type_score_df = pd.DataFrame(gene_type_score_dict)

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        simulated_clone_assignment_format = output_dir + "/simulated_clone_assignment_gene_{gene_count}_snp_{snp_count}_cell_{cell_count}_cnv_{cnv_freq}_index_{index}.csv"
        simulated_gene_type_score_format = output_dir + "/simulated_gene_type_score_gene_{gene_count}_snp_{snp_count}_cell_{cell_count}_cnv_{cnv_freq}_index_{index}.csv"
        simulated_expr_format = output_dir + "/simulated_expr_gene_{gene_count}_snp_{snp_count}_cell_{cell_count}_cnv_{cnv_freq}_index_{index}.csv"
        simulated_cnv_format = output_dir + "/simulated_cnv_gene_{gene_count}_snp_{snp_count}_cell_{cell_count}_cnv_{cnv_freq}_index_{index}.csv"

        if self.generate_allelic_data:
            # sample snps
            snp_ids = random.choices(range(self.hscn.shape[0]), k=snp_count)
            # snp_ids = random.sample(range(self.snv.shape[0]), k=snp_count)
            simulated_snv_allele_format = output_dir + "/simulated_snv_allele_gene_{gene_count}_snp_{snp_count}_cell_{cell_count}_cnv_{cnv_freq}_index_{index}.csv"
            simulated_snv_format = output_dir + "/simulated_snv_gene_{gene_count}_snp_{snp_count}_cell_{cell_count}_cnv_{cnv_freq}_index_{index}.csv" 
            simulated_hscn_format = output_dir + "/simulated_hscn_gene_{gene_count}_snp_{snp_count}_cell_{cell_count}_cnv_{cnv_freq}_index_{index}.csv"
        else:
            snp_ids = None           

        for i in range(len(cnv_dependency_freqs)):
            for j in range(len(cell_samples)):
                current_freq = cnv_dependency_freqs[i]
                cell_sample = cell_samples[j]

                simulated_cell_assignment, gene_type_score_simulated, expected_expr_df, simulated_snv, simulated_snv_allele = self.simulate_individual_data(
                    gene_ids, snp_ids, cell_sample, gene_type_score_df[current_freq], add_snp_read_count_prob, add_noises_per_cell)
                
                simulated_cnv_df = self.clone_cnv_df.iloc[gene_ids, cell_sample]
                
                simulated_cnv_df.columns = range(len(cell_sample))

                simulated_cell_assignment.to_csv(
                    simulated_clone_assignment_format.format(cell_count=len(cell_sample), cnv_freq=current_freq, gene_count=gene_count, index=index, snp_count=snp_count))

                gene_type_score_simulated.to_csv(
                    simulated_gene_type_score_format.format(cell_count=len(cell_sample), cnv_freq=current_freq, gene_count=gene_count, index=index, snp_count=snp_count))

                expected_expr_df.to_csv(
                    simulated_expr_format.format(cell_count=len(cell_sample), cnv_freq=current_freq, gene_count=gene_count, index=index, snp_count=snp_count))

                simulated_cnv_df.to_csv(simulated_cnv_format.format(cell_count=len(cell_sample), cnv_freq=current_freq, gene_count=gene_count, index=index, snp_count=snp_count))

                if self.generate_allelic_data:
                    # reindex
                    simulated_hscn = self.hscn.iloc[snp_ids, cell_sample].reset_index(drop=True)
                    simulated_hscn.columns = simulated_snv.columns
                    simulated_snv.reset_index(drop=True).to_csv(simulated_snv_format.format(cell_count=len(cell_sample), cnv_freq=current_freq, gene_count=gene_count, index=index, snp_count=snp_count))

                    simulated_snv_allele.reset_index(drop=True).to_csv(simulated_snv_allele_format.format(cell_count=len(cell_sample), cnv_freq=current_freq, gene_count=gene_count, index=index, snp_count=snp_count))
                    simulated_hscn.to_csv(simulated_hscn_format.format(cell_count=len(cell_sample), cnv_freq=current_freq, gene_count=gene_count, index=index, snp_count=snp_count))
                

    def simulate_individual_data(self, gene_ids, snp_ids, cell_sample, gene_type_score, add_snp_read_count_prob, add_noises_per_cell):

        current_cnv_df = self.clone_cnv_df.iloc[gene_ids, ]
        current_cnv = torch.tensor(current_cnv_df.values).transpose(0, 1)

        per_copy_expr = self.map_estimates['expose_per_copy_expr'][gene_ids]
        w = self.map_estimates['expose_w'][gene_ids]

        psi = self.map_estimates['expose_psi']
        psi = psi[random.choices(range(psi.shape[0]), k=len(cell_sample))]

        softplus = Softplus()

        # calculate copy number mean
        per_copy_expr = softplus(per_copy_expr)

        gene_type_score = torch.tensor(gene_type_score.values)

        if self.generate_allelic_data:
            current_hscn_df = self.hscn.iloc[snp_ids, cell_sample]        
            current_hscn = torch.tensor(current_hscn_df.values).transpose(0, 1)


        with pyro.plate('cell', len(cell_sample)):
            expected_expr = (per_copy_expr * Vindex(current_cnv)[cell_sample] * gene_type_score +
                             per_copy_expr * (1 - gene_type_score)) * \
                            torch.exp(torch.matmul(psi, torch.transpose(w, 0, 1)))

            # draw expr from Multinomial
            expr_simulated = pyro.sample('expr',
                                         dist.Multinomial(total_count=10000, probs=expected_expr, validate_args=False))
            # add more noise
            noise_prob = pyro.sample('noise_prob', dist.Dirichlet(torch.zeros(expr_simulated.shape[0], expr_simulated.shape[1]) + 1))
            expr_simulated = expr_simulated + pyro.sample('noise_count', dist.Multinomial(total_count = add_noises_per_cell, probs = noise_prob, validate_args=False))


        # simulate allele specific data
        if self.generate_allelic_data:
            # fake snv total
            current_snv_df = self.snv.iloc[snp_ids, ]
            current_snv = torch.tensor(current_snv_df.values).transpose(0, 1)

            # random select snv total profile
            indices = torch.tensor(random.choices([_ for _ in range(len(current_snv))], k=len(cell_sample)))
            current_snv = current_snv[indices]

            # add random number to increase snv read counts
            rand_snv = Binomial(1, torch.rand(current_snv.shape[0], current_snv.shape[1]) * add_snp_read_count_prob)
            current_snv = current_snv + rand_snv.sample()

            # generate snv_allele
            snv_allele_simulated = pyro.sample('hscn', dist.Binomial(current_snv, current_hscn))


        expected_expr_df = pd.DataFrame(expr_simulated.transpose(0, 1).detach().numpy())
        expected_expr_df.index = current_cnv_df.index.values

        gene_type_score_simulated = pd.DataFrame(
            {'gene': expected_expr_df.index.values, 'gene_type_score': gene_type_score.detach().numpy()})

        cell_sample_names = self.clone_cnv_df.columns.values[cell_sample].tolist()
        simulated_cell_assignment = pd.DataFrame(
            {'cell_id': expected_expr_df.columns.values.tolist(), 'clone_id': cell_sample_names})
        
        if self.generate_allelic_data:
            expected_snv = pd.DataFrame(current_snv.transpose(0, 1).detach().numpy())
            expected_snv_allele = pd.DataFrame(snv_allele_simulated.transpose(0, 1).detach().numpy())
        else:
            expected_snv = None
            expected_snv_allele = None


        return simulated_cell_assignment, gene_type_score_simulated, expected_expr_df, expected_snv, expected_snv_allele
