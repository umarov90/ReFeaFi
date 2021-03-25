import sys
import math

input_file = sys.argv[1]
output_file = sys.argv[2]

bed_lines = []
with open(input_file) as file:
    for line in file:
        vals = line.split("\t")
        score = float(vals[5])
        chrn  = vals[0]
        strand = vals[6]
        score = math.log(1 + score / (1.00001 - score))
        pos = int(int(vals[3]) + (int(vals[4]) - int(vals[3])) / 2) - 1
        col = "255,0,0"
        if strand == "-":
            col = "0,0,255"
        elif strand == "+":
            col = "0,255,0"
        entry = chrn + "\t" + vals[3] + "\t" + vals[4] + "\t"\
                + chrn + ":" +  vals[3] + ":" + vals[4] + ":" + strand\
                + "\t" + str(score) + "\t" + strand + "\t" + str(pos) + "\t" + str(pos + 1) + "\t" + col
        bed_lines.append(entry)

with open(output_file, 'w+') as f:
    f.write('\n'.join(bed_lines))