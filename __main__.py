import argparse
from src import process_input_matrices, process_output_matrices, run_clonealign_pyro

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_suffix(x, suffix):
    x_elements = x.split('.')
    x_elements.insert(len(x_elements) - 1, str(suffix))
    x_path = '.'.join(x_elements)
    return x_path

def main():
    parser = argparse.ArgumentParser(description='clonealign inputs: expr matrix, gene * clone cnv matrix')
    parser.add_argument('-e', '--expr', nargs=1, help='expr matrix')
    parser.add_argument('-c', '--clone', nargs=1, help='gene * clone cnv matrix')
    parser.add_argument('-a', '--assignment', nargs=1, help='output path for clone assignment')
    parser.add_argument('-s', '--gene_score', nargs=1, default='', help='output path for gene type scores')
    parser.add_argument('-f', '--fold_change', nargs=1, default='', help='output path for gene fold change')
    parser.add_argument('-r', '--repeat_times', type=int, default=1, help="number of times of repeating the clonealign run")
    parser.add_argument('-g', '--gene_mode', nargs=1, default='default_model', help='which model to use')
    args = parser.parse_args()

    expr, cnv, expr_csv, cnv_csv = process_input_matrices(args.expr[0], args.clone[0])

    repeat_times = int(args.repeat_times)
    for i in range(repeat_times):
        print(args.gene_mode)
        clone_assign_prob, gene_type_score, gene_fold_change = run_clonealign_pyro(cnv, expr, args.gene_mode[0])
        if i == 0:
            print("output the first results!")
            process_output_matrices(clone_assign_prob, gene_type_score, gene_fold_change, args.assignment[0], args.gene_score[0], args.fold_change[0], expr_csv, cnv_csv)
        else:
            assignment_path = add_suffix(args.assignment[0], i)
            gene_score_path = add_suffix(args.gene_score[0], i)
            fold_change_path = add_suffix(args.fold_change[0], i)
            process_output_matrices(clone_assign_prob, gene_type_score, gene_fold_change, assignment_path, gene_score_path, fold_change_path, expr_csv, cnv_csv)

    print("clonealign pyro is finished!")


if __name__ == "__main__":
    main()
