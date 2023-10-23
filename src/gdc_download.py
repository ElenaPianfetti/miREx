"""Downloads the data from the GDC portal.
For this script to work manifests and json files must be present in the data directory (<data_dir>/<primary_site>).)"""

import os
import json
import argparse


def check_dirs(data_d, canc):
    """Checks if the correct directories and files exist for the given primary site.
    
    Args:
        data_d (str): Directory where the data is stored for each primary site
        canc (str): Primary site
    """
    c_dir = os.path.join(data_d, canc)
    # check if the primary site directory exists
    if canc not in os.listdir(data_d):
        raise Exception(f"Missing {canc} directory\n")
    if 'miRNA' not in os.listdir(c_dir):
        # create miRNA directory
        os.mkdir(os.path.join(c_dir, 'miRNA'))
    if 'mRNA' not in os.listdir(c_dir):
        # create mRNA directory
        os.mkdir(os.path.join(c_dir, 'mRNA'))
    # check if there is a manifest and json files
    for omic in args.omics:
        if f'{canc}_{omic}.txt' not in os.listdir(c_dir):
            raise Exception(f"Missing {canc}_{omic}.txt\n")

        if f'{canc}_{omic}.json' not in os.listdir(c_dir):
            raise Exception(f"Missing {canc}_{omic}.json\n")


def create_new_manifest(data_d, canc):
    """Creates a new manifest for the given primary site, with only the files that have both mRNA and miRNA data.
    
    Args:
        data_d  (str): Directory where the data is stored for each primary site
        canc    (str): Primary site

    Returns:
        n_cases (int): Number of cases that have both mRNA and miRNA data
    """
    
    # read the json files
    with open(os.path.join(data_d, canc, f'{canc}_miRNA.json'), 'r') as f:
        miRNA_data = json.load(f)
    with open(os.path.join(data_d, canc, f'{canc}_mRNA.json'), 'r') as f:
        mRNA_data = json.load(f)

    # find all cases
    miRNA_cases = [case['cases'][0]['case_id'] for case in miRNA_data]
    mRNA_cases = [case['cases'][0]['case_id'] for case in mRNA_data]

    # if one case appears more than once, only keep one
    miRNA_data = list({case['cases'][0]['case_id']: case for case in miRNA_data}.values())
    mRNA_data = list({case['cases'][0]['case_id']: case for case in mRNA_data}.values())

    common_cases = list(set(miRNA_cases).intersection(set(mRNA_cases)))
    n_cases = len(common_cases)

    # remove the cases that are not common
    miRNA_files = [case['file_name'] for case in miRNA_data if case['cases'][0]['case_id'] in common_cases]
    mRNA_files = [case['file_name'] for case in mRNA_data if case['cases'][0]['case_id'] in common_cases]

    # read the manifests
    with open(os.path.join(data_d, canc, f'{canc}_mRNA.txt'), 'r') as f:
        mRNA_manifest = f.readlines()
        mRNA_header = mRNA_manifest[0]
    with open(os.path.join(data_d, canc, f'{canc}_miRNA.txt'), 'r') as f:
        miRNA_manifest = f.readlines()
        miRNA_header = miRNA_manifest[0]
    
    # only keep the files that are common
    mRNA_manifest = [line for line in mRNA_manifest if line.split('\t')[1] in mRNA_files]
    mRNA_manifest.insert(0, mRNA_header)
    miRNA_manifest = [line for line in miRNA_manifest if line.split('\t')[1] in miRNA_files]
    miRNA_manifest.insert(0, miRNA_header)

    # write the new manifests
    with open(os.path.join(data_d, canc, f'{canc}_mRNA_manifest.txt'), 'w') as f:
        f.writelines(mRNA_manifest)
    with open(os.path.join(data_d, canc, f'{canc}_miRNA_manifest.txt'), 'w') as f:
        f.writelines(miRNA_manifest)

    return n_cases


def main(args):

    for primary_site in args.primary_sites:
        print(f"primary site: {primary_site}")

        # check if the correct directories and files exist
        check_dirs(args.data_dir, primary_site)
        
        # create new manifests with only the files that have both mRNA and miRNA data
        n_files = create_new_manifest(args.data_dir, primary_site)

        for omic in args.omics:
            if len(os.listdir(os.path.join(args.data_dir, primary_site, omic))) == n_files:
                print(f"{omic} data already downloaded")
                continue
            # the command for the download is ./gdc-client download -m <manifest> -d <output directory>
            # the manifest is the <primary_site>_<omic>_manifest.txt file
            # the output directory is the <primary_site>/<omic> directory
            print(f"Downloading {omic} data")
            call = f"./gdc-client download -m {os.path.join(args.data_dir, primary_site, f'{primary_site}_{omic}_manifest.txt')} -d {os.path.join(args.data_dir, primary_site, omic)}"
            print(call)
            os.system(call)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/gdc_data', help='Directory where the data is stored for each primary site')
    parser.add_argument('--omics', type=list, default=['mRNA', 'miRNA'], help='List of omics to download, for which json and manifest files exist')
    parser.add_argument('--primary_sites', type=list, default=['lung', 'breast', 'kidney', 'uterus'], help='List of primary sites to download')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    args = parser.parse_args()
    
    if args.verbose:
        print('\nStarting with arguments:')
        # get longest argument name
        max_len = max([len(arg) for arg in vars(args)])
        for arg in vars(args):
            print(f"{arg:{max_len}} : {getattr(args, arg)}")
        print('\n')
    
    main(args)
            