import argparse
from process_input import process_input
from get_parameters_from_real_data import get_parameters, clonealign_pyro_simulation
import torch
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='clonealign inputs: expr matrix, gene * clone cnv matrix')
    parser.add_argument('-e', '--expr', nargs=1, help='expr matrix')
    parser.add_argument('-c', '--clone', nargs=1, help='gene * clone cnv matrix')
    parser.add_argument('-g', '--gene_type_freqs', action='append',
                        help='the frequency of gene_type_freq equaling zero')
    parser.add_argument('-o', '--output_dir', nargs=1, help='output directory')

    args = parser.parse_args()
    output_dir = args.output_dir[0]

    expr_path, cnv_path, gene_type_freqs = args.expr[0], args.clone[0], args.gene_type_freqs

    expr, cnv, expr_csv, cnv_csv= process_input(expr_path, cnv_path)

    cnv, expr, per_copy_expr, mean_expr, psi, w = get_parameters(expr, cnv)

    for gene_type_freq in gene_type_freqs:
        expr_simulated, gene_type_score_simulated, clone_assign_simulated = clonealign_pyro_simulation(cnv, expr,
                                                                                                       per_copy_expr,
                                                                                                       mean_expr,
                                                                                                       psi, w,
                                                                                                       int(gene_type_freq))

        expr_simulated = torch.transpose(expr_simulated, 0, 1)

        expr_simulated_dataframe = pd.DataFrame(expr_simulated.data.numpy())
        gene_type_score_simuated_dataframe = pd.DataFrame(gene_type_score_simulated.data.numpy())
        clone_assign_simulated_dataframe = pd.DataFrame(clone_assign_simulated.data.numpy())

        # rename
        cell_name = {i: c for i, c in enumerate(expr_csv.columns)}
        gene_name = {i: c for i, c in enumerate(expr_csv.index)}
        clone_name = {i: c for i, c in enumerate(cnv_csv.columns)}

        expr_simulated_dataframe.rename(index=gene_name, inplace=True)
        expr_simulated_dataframe.rename(columns=cell_name, inplace=True)

        gene_type_score_simuated_dataframe.rename(index=gene_name, inplace=True)
        clone_assign_simulated_dataframe.rename(index=cell_name, inplace=True)

        expr_simulated_dataframe.to_csv(output_dir + "/expr_simulated_" + str(gene_type_freq) + ".csv")
        gene_type_score_simuated_dataframe.to_csv(output_dir + "/gene_type_score_simulated_" + str(gene_type_freq) + ".csv")
        clone_assign_simulated_dataframe.to_csv(output_dir + "/clone_assign_simulated_" + str(gene_type_freq) + ".csv")

    print("simulation finished")


if __name__ == "__main__":
    main()
