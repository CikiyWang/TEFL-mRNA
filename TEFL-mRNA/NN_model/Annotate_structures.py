# scripts for annotate structures from LinearFold results
# 
# Input format:
#	 '>header
#    'CCCCAUAGGGG
#    '((((...)))) (-3.3)
#    This is the default format of e.g. RNAfold/LinearFold. The output file will contain the annotated string:
# Output format:
#    '>header
#    'CCCCAUAGGGG
#    'SSSSHHHSSSS
# gzip *.fa

# Usage:
# 

import os
import sys
import forgi.graph.bulge_graph as fgb
import gzip
from itertools import groupby, repeat

def get_handle(file_name, mode):
    if file_name[-2:] == "gz":
        return gzip.open(file_name, mode)
    return open(file_name, mode)


def parse_fasta(handle, joiner = ""):
    delimiter = lambda line: line.startswith('>')
    for is_header, block in groupby(handle, delimiter):
        if is_header:
            header = next(block)[1:].rstrip()
        else:
            yield(header, joiner.join(line.rstrip() for line in block))


def annotate_structures(input_file, output_file):
    """ Annotate secondary structure predictions with structural contexts.
    Given dot-bracket strings this function will annote every character as
    either 'H' (hairpin), 'S' (stem), 'I' (internal loop/bulge), 'M' (multi loop), 'F' (5-prime)
    or 'T' (3-prime). The input file must be a fasta formatted file and each sequence and structure
    must span a single line:
    '>header
    'CCCCAUAGGGG
    '((((...)))) (-3.3)
    This is the default format of e.g. RNAfold/LinearFold. The output file will contain the annotated string:
    '>header
    'CCCCAUAGGGG
    'SSSSHHHSSSS
    Parameters
    ----------
    input_file : str
        A fasta file containing secondary structure predictions.
    
    output_file : str
        A fasta file with secondary structure annotations.
    """
    handle_in = get_handle(input_file, "rt")
    handle_out = get_handle(output_file, "wt")
    for header, entry in parse_fasta(handle_in, "_"):
        entry = entry.split("_")
        bg = fgb.BulgeGraph.from_dotbracket(entry[1].split()[0])
        handle_out.write(">{}\n".format(header))
        handle_out.write("{}\n{}\n".format(entry[0].replace('T', 'U'), bg.to_element_string().upper()))
    handle_in.close()
    handle_out.close()


in_file = sys.argv[1] # "/picb/rnasys2/wangsiqi/Codon_opt/Ribo_seq/LinearFold_Res/TE_inter_low0.25_NM_ids_CDS_sequences.linearfold"
out_file = sys.argv[2] # "/picb/rnasys2/wangsiqi/Codon_opt/Ribo_seq/LinearFold_Res/TE_inter_low0.25_NM_ids_CDS_sequences.linearfold_anno.fa"

# run annotation process
annotate_structures(in_file,out_file)
