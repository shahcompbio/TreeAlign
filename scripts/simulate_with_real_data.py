import argparse
import os
from process_input import process_input
from get_parameters_from_real_data import get_parameters, clonealign_pyro_simulation, cnv_simulation
import torch
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='clonealign inputs: expr matrix, gene * clone cnv matrix')
    parser.add_argument('-e', '--expr', nargs=1, help='expr matrix')
    parser.add_argument('-c', '--clone', nargs=1, help='gene * clone cnv matrix')
    parser.add_argument('-x', '--clone_count', nargs=1, default='2')
    parser.add_argument('-g', '--gene_type_freqs', action='append',
                        help='the frequency of gene_type_freq equaling zero', default=["50"])
    parser.add_argument('-k', '--cell_counts', action='append',
                        help='the number of cells', default=["1000"])
    parser.add_argument('-n', '--gene_counts', action='append',
                        help='the number genes',  default=["1000"])
    parser.add_argument('-o', '--output_dir', nargs=1, help='output directory')

    args = parser.parse_args()
    output_dir = args.output_dir[0]
    clone_count = int(args.clone_count[0])

    expr_path, cnv_path = args.expr[0], args.clone[0]

    gene_type_freqs = args.gene_type_freqs
    cell_counts = args.cell_counts
    gene_counts = args.gene_counts

    expr, cnv, expr_csv, cnv_csv = process_input(expr_path, cnv_path)

    per_copy_expr, mean_expr, psi, w = get_parameters(expr, cnv)

    for gene_type_freq in gene_type_freqs:
        for cell_count in cell_counts:
            for gene_count in gene_counts:
                cnv_simulated = cnv_simulation(clone_count, int(gene_count))
                expr_simulated, gene_type_score_simulated, clone_assign_simulated, random_cells, random_genes = clonealign_pyro_simulation(cnv_simulated, expr,
                                                                                                               per_copy_expr,
                                                                                                               psi, w,
                                                                                                               int(gene_type_freq),
                                                                                                               int(cell_count),
                                                                                                               int(gene_count))

                expr_simulated = torch.transpose(expr_simulated, 0, 1)
                cnv_simulated = torch.transpose(cnv_simulated, 0, 1)

                expr_simulated_dataframe = pd.DataFrame(expr_simulated.data.numpy())
                cnv_simulated_dataframe = pd.DataFrame(cnv_simulated.data.numpy())
                gene_type_score_simulated_dataframe = pd.DataFrame(gene_type_score_simulated.data.numpy())
                clone_assign_simulated_dataframe = pd.DataFrame(clone_assign_simulated.data.numpy())

                # rename
                cell_name = {i: c for i, c in enumerate(expr_csv.columns[random_cells])}
                gene_name = {i: c for i, c in enumerate(expr_csv.index[random_genes])}

                expr_simulated_dataframe.rename(index=gene_name, inplace=True)
                expr_simulated_dataframe.rename(columns=cell_name, inplace=True)
                cnv_simulated_dataframe.rename(index=gene_name, inplace=True)

                gene_type_score_simulated_dataframe.rename(index=gene_name, inplace=True)
                clone_assign_simulated_dataframe.rename(index=cell_name, inplace=True)

                output_string = str(gene_type_freq) + "_"  + str(cell_count) + "_" + str(gene_count)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                expr_simulated_dataframe.to_csv(output_dir + "/expr_simulated_" + output_string + ".csv")
                cnv_simulated_dataframe.to_csv(output_dir + "/cnv_simulated_" + output_string + ".csv")
                gene_type_score_simulated_dataframe.to_csv(output_dir + "/gene_type_score_simulated_" + output_string + ".csv")
                clone_assign_simulated_dataframe.to_csv(output_dir + "/clone_assign_simulated_" + output_string + ".csv")

    print("simulation finished")


if __name__ == "__main__":
    main()
