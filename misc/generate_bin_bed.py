bed = []
bin = 192
for i in range(1, 249250621 - bin, bin):
    bed.append("chr1" + "\t" + str(i) + "\t" + str(i + bin) + "\t" + "x" + "\t" + "0" + "\t" + ".")

with open("bin.bed", 'w+') as f:
    f.write('\n'.join(bed))

