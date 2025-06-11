# python script for integrate LinearFold structure annotation results
# python3
# usage: python trans_structure_annot_res_totsv.py prefix_name

# Input: results from Annotate_structures.py 
# 	Annotate_structures output: 
# 		Line 1: names;
# 		Line 2: sequecnces;
# 		Line 3: structures;
# 		Line 4: annotated structures;

# Output: 
# 	table in columns: *_linearfold_anno_res.tsv
# 		Column 1: names;
# 		Column 2: sequences;
#		Column 3: structure;
#		Column 3: structure annotation;

# easy to do downstream analysis in loops

import pandas as pd 
import numpy as py
import sys

# read result file name for command
file_in = sys.argv[1] + ".linearfold_anno.fa"
file_out = sys.argv[1] + "_linearfold_anno_res.tsv"
print(file_in)
print(file_out)

file_ins = open(file_in)
file_outs = open(file_out,'w')

res = "transcript_refid" + "\t" + "Sequence_FL" + "\t" + "Structure_anno_FL" + "\n"

# read in lines
# ">" --> title
# ">" + 1 --> sequence
# ">" + 2 --> remove '(\d+)' in tails --> structure
# ">" + 2 --> '(\d+)' in tails --> MFE
for line in file_ins:
	if line[0] == '>':
		name = line.strip('\n')
		#print(name)
		transcript_refid = name.split("|")[3]
		res = res + transcript_refid + "\t"
	elif line[0] in ['A','G','C','U']:
		seq = line.strip('\n')
		res = res + seq + "\t"
	else:
		structure_anno = line.strip('\n')
		res = res + structure_anno + "\n"

print(res, file = file_outs)
file_outs.close()


