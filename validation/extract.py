from Bio.Seq import Seq

def extract(folder, gn):
    seq = ""
    print(folder+"/" + gn)
    with open(folder+"/" + gn) as f:
        for line in f:
            if(line.startswith(">")):
                if(len(seq)!=0):                    
                    break
                chrn = line.strip()[1:]             
                seq = ""
                continue                
            else:
                seq+=line
    print(len("".join(seq.split())))
    with open("chr1/" + folder + "_chr1.fa", 'w+') as f:
        f.write(">chr1")
        f.write("\n")
        f.write(seq.strip())
    with open("chr1/" + folder + "_chr1_rev.fa", 'w+') as f:
        f.write(">chr1")
        f.write("\n")
        seq = Seq(seq)
        f.write(str(seq.reverse_complement()).strip())

extract("human", "hg19.fa")
extract("mouse", "chr1.fa")
extract("rat", "rn6.fa") 
extract("chicken", "galGal5.fa")
extract("dog", "canFam3.fa") 
extract("monkey", "rheMac8.fa")


