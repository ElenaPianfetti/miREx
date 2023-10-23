import argparse
import os


def parse_strings(arg):
    return arg.split(',')

def parse_and_check():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='data/h5_datasets', type=str, help='Directory where the datasets are stored')
    parser.add_argument('--resultsdir', default='results', type=str, help='Directory where the results will be stored')
    parser.add_argument('--cancer_subtype', type=str, help='Subtype of tumor', required=True)
    parser.add_argument('--primary_site', type=str, help='Cancer type', required=True)
    parser.add_argument('--mir', type=int, choices=[0, 1], help='Whether to use miRNA data or not')
    parser.add_argument('--mir_type', default='all', type=str, help='Which list of mirnas to use')
    parser.add_argument('--verbose', type=int, choices=[0, 1], default=1, help='Print more output')
    parser.add_argument('--n_train', default=20, type=int, help='Number of different models to train')
    parser.add_argument('--Xpresso_split', default=1, type=int, help='Use Xpresso split or not')
    parser.add_argument('--genes_dir', type=str, default='data/genes', help='Directory where the genes are stored')
    parser.add_argument('--old_dataset', type=int, choices=[0, 1], default=0, help='Use old dataset or not')
    args = parser.parse_args()

    # create results directory
    if args.mir:
        if args.Xpresso_split:
            results_dir = os.path.join(args.resultsdir, f"{args.primary_site}_Xp", 'mirna', args.mir_type, args.cancer_subtype)
        else:
            results_dir = os.path.join(args.resultsdir, f"{args.primary_site}_cv", 'mirna', args.mir_type, args.cancer_subtype)
    else:
        if args.Xpresso_split:
            results_dir = os.path.join(args.resultsdir, f"{args.primary_site}_Xp", 'no_mirna', args.cancer_subtype)
        else:
            results_dir = os.path.join(args.resultsdir, f"{args.primary_site}_cv", 'no_mirna', args.cancer_subtype)
    args.resultsdir = results_dir

    # check if results directory already exists
    if os.path.exists(args.resultsdir):
        raise Exception(f"Results directory {args.resultsdir} already exists. Please delete it or change the name of the results directory to avoid overwriting results.")
    else:
        os.makedirs(args.resultsdir)
        for i in range(args.n_train):
            os.makedirs(os.path.join(args.resultsdir, str(i)))
            os.makedirs(os.path.join(args.resultsdir, str(i), 'split'))
    # check if inputs exist
    if not os.path.exists(os.path.join(args.datadir, "sequence_data.h5")):
        raise Exception(f"File {os.path.join(args.datadir, 'sequence_data.h5')} does not exist.")
    if not os.path.exists(os.path.join(args.datadir, f"{args.primary_site}.h5")):
        raise Exception(f"File {os.path.join(args.datadir, f'{args.primary_site}.h5')} does not exist.")

    if args.mir:
        if args.mir_type != "all":
            with open(f"mirna_lists/{args.mir_type}_{args.primary_site}_{args.cancer_subtype}.txt") as file:
                mirna_list = file.readlines()
                args.mirna_list = [mirna.strip() for mirna in mirna_list]
            print(f"Loading mirna_lists/{args.mir_type}_{args.primary_site}_{args.cancer_subtype}.txt")
        else:
            args.mirna_list = False
    else:
        args.mirna_list = False


    if args.verbose:
        print('\nStarting with arguments:')
        # get longest argument name
        max_len = max([len(arg) for arg in vars(args)])
        for arg in vars(args):
            print(f"{arg:{max_len}} : {getattr(args, arg)}")
        print('\n')

    return args