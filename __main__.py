import argparse
from src import process_input_matrices, process_output_matrices, run_clonealign_pyro


def main():
    parser = argparse.ArgumentParser(description='clonealign inputs: expr matrix, gene * clone cnv matrix')
    parser.add_argument('-e', '--expr', nargs=1, help='expr matrix')
    parser.add_argument('-c', '--clone', nargs=1, help='gene * clone cnv matrix')
    parser.add_argument('-a', '--assignment', nargs=1, help='output path for clone assignment')
    parser.add_argument('-s', '--gene_score', nargs=1, help='output path for gene type scores')
    args = parser.parse_args()

    print(args.expr)
    expr, cnv = process_input_matrices(args.expr[0], args.clone[0])

    clone_assign_prob, gene_type_score = run_clonealign_pyro(cnv, expr)

    process_output_matrices(clone_assign_prob, gene_type_score, args.assignment[0], args.gene_score[0], expr, cnv)
    print("clonealign pyro is finished!")


if __name__ == "__main__":
    main()
