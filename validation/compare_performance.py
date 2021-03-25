import os
import numpy as np
import math

os.chdir(open("../data_dir").read().strip())

gl = {}
gl["human"] = 249250621
gl["mouse"] = 197195432
gl["chicken"] = 196202544
gl["monkey"] = 225584828
gl["rat"] = 282763074
gl["dog"] = 122678785

def get_prompredict(organism):
    preds = []
    dt = 2.0
    mvs = []
    with open("data/competitors/" + organism + "_chr1_PPde.txt") as file:
        for line in file:
            if(line.startswith("#")):
                continue
            vals = line.split("\t")
            if(float(vals[10]) > dt):
                p = (int(vals[2]) + int(vals[3])) / 2.0
                preds.append([p, float(vals[10])])
            mvs.append(float(vals[10]))

    with open("data/competitors/" + organism + "_chr1_rev_PPde.txt") as file:
        for line in file:
            if(line.startswith("#")):
                continue
            vals = line.split("\t")
            p = (int(vals[2]) + int(vals[3])) / 2.0
            p = gl[organism] - p
            preds.append([p, float(vals[10])])
            mvs.append(float(vals[10]))

    #print(max(mvs))
    #print(min(mvs))
    preds.sort()
    #print("PromPredict predictions: " + str(len(preds["+"]) + len(preds["-"])))
    return preds

def get_ep3(organism):
    preds = []
    mvs = []
    with open("data/competitors/" + organism + "_chr1.fa.gff3") as file:
        for line in file:
            vals = line.split("\t")
            p = int(vals[3]) + 200 
            preds.append([p, float(vals[5])])
            mvs.append(float(vals[5]))

    with open("data/competitors/" + organism + "_chr1_rev.fa.gff3") as file:
        for line in file:
            vals = line.split("\t")
            p = int(vals[3]) + 200   
            p = gl[organism] - p    
            preds.append([p, float(vals[5])])
            mvs.append(float(vals[5]))

    #print(max(mvs))
    #print(min(mvs))
    preds.sort()
    #print("EP3 predictions: " + str(len(preds["+"]) + len(preds["-"])))
    return preds

def get_deeprefind(fpath, type):
    min_score = 100
    max_score = -100
    preds = []
    with open(fpath) as file:
        for line in file:
            vals = line.split("\t")
            score = float(vals[5])
            # if vals[6] == ".":
            #     if type == "promoter":
            #         continue
            try:
                score = math.log(1 + score / (1.00001 - score))
                if score < min_score:
                    min_score = score
                if score > max_score:
                    max_score = score
                if(vals[2] == "promoter/enhancer"):
                    pos = int(int(vals[3]) + (int(vals[4]) - int(vals[3])) / 2) - 1
                    preds.append([pos, score])
            except:
                pass
    # preds.sort()
    return preds


def get_results(organism, path):
    deepag = get_deeprefind(organism + "_chr1.gff", type="promoter")
    cage = get_cage(path, organism)
    print(organism + " : " + str(len(cage["chr1"])))

    print("deeprag")
    dtv = []
    for dti in range(100):
        dt = -2.2 + dti*0.123
        res = compare(cage, deepag, dt)
        fpr = 1000000 * (res[0]/(2*gl[organism]))
        dtv.append(str(res[1]) + "," + str(fpr))
        if round(res[1], 2) == 0.50:
            best_line = str(res[1]) + "\t" + str(res[2]) + "\t" + str(res[3]) + "\t" + str(res[4]) + "\t" + str(fpr)
            print(best_line)

    with open("figures_data/dtv_deeprag_"+organism+".csv", 'w+') as f:
        for l in dtv:
            f.write(str(l) + "\n")

    if organism != "human":
        return
    ep3 = get_ep3(organism)
    prompredict = get_prompredict(organism)
    basenji = get_basenji()
    print("ep3")
    dtv = []
    for dti in range(100):
        dt = -0.25 + dti * 0.011
        res = compare(cage, ep3, dt)
        fpr = 1000000 * (res[0]/(2*gl[organism]))
        dtv.append(str(res[1]) + "," + str(fpr))
        if round(res[1], 2) == 0.50:
            best_line = str(res[1]) + "\t" + str(res[2]) + "\t" + str(res[3]) + "\t" + str(res[4]) + "\t" + str(fpr)
            print(best_line)
    with open("figures_data/dtv_ep3_"+organism+".csv", 'w+') as f:
        for l in dtv:
            f.write(str(l) + "\n")


    print("basenji")
    dtv = []
    for dti in range(160):
        dt = 0.6 + dti * 0.0025
        res = compare(cage, basenji, dt)
        fpr = 1000000 * (res[0] / (2 * gl[organism]))
        dtv.append(str(res[1]) + "," + str(fpr))
        if round(res[1], 2) == 0.50:
            best_line = str(res[1]) + "\t" + str(res[2]) + "\t" + str(res[3]) + "\t" + str(res[4]) + "\t" + str(fpr)
            print(best_line)

    with open("figures_data/dtv_basenji7_" + organism + ".csv", 'w+') as f:
        for l in dtv:
            f.write(str(l) + "\n")

    print("prompredict")
    dtv = []
    for dti in range(100):
        dt = 2.8 + dti * 0.032
        res = compare(cage, prompredict, dt)
        fpr = 1000000 * (res[0]/(2*gl[organism]))
        dtv.append(str(res[1]) + "," + str(fpr))
        if round(res[1], 2) == 0.50:
            best_line = str(res[1]) + "\t" + str(res[2]) + "\t" + str(res[3]) + "\t" + str(res[4]) + "\t" + str(fpr)
            print(best_line)
    with open("figures_data/dtv_prompredict_"+organism+".csv", 'w+') as f:
        for l in dtv:
            f.write(str(l) + "\n")


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def compare(cage, preds, dt, margin=500):
    tp = 0.0
    fn = 0.0
    fp = 0.0
    key = "chr1"
    cg = [i[0] for i in cage[key]]
    cg = np.asarray(cg)
    pr = [i[0] for i in preds if i[1] >= dt]
    pr.sort()
    pr = np.asarray(pr)
    if(len(pr) > 0):
        for gt in cg:
            v = find_nearest(pr, gt)
            if(abs(v - gt) <= margin):
                tp = tp + 1                
            else:
                fn = fn + 1

    for v in pr:
        gt = find_nearest(cg, v)
        if(abs(v - gt) > margin):                                 
            fp = fp + 1
    if(tp > 0):
        recall = tp / (tp + fn) 
        precision = tp / (tp + fp)
    else:
        recall = 0  
        precision = 0 
    if( (recall + precision) != 0):
        f1 = 2.0 * ( (recall*precision) / (recall + precision) )
    else:
        f1 = 0

    if tp > 0:
        div = fp / tp
    else:
        div = 0
    return fp, recall, precision, f1, div


def get_basenji():
    reg_elements = []
    positions = []
    bin = 192
    for i in range(1, 249250621 - bin, bin):
        positions.append(i + bin / 2)
    k = 0
    with open("data/competitors/basenji_scores.txt") as file:
        for line in file:
            val = float(line)
            reg_elements.append([positions[k], val])
            k = k + 1
    return reg_elements


def get_cage(path, organism):
    reg_elements = {}
    with open(path) as file:
        for line in file:
            vals = line.split("\t")
            chrn = vals[0]
            chrp = int(vals[7]) - 1
            val = 1
            reg_elements.setdefault(chrn,[]).append([chrp, val])
    if organism == "human":
        with open("data/human_permissive_enhancers_phase_1_and_2.bed") as file:
            for line in file:
                vals = line.split("\t")
                chrn = vals[0]
                val = 1
                chrp = int(vals[7]) - 1
                reg_elements.setdefault(chrn,[]).append([chrp, val])
    for key in reg_elements.keys():
        reg_elements[key].sort(key=lambda x: x[0])
    return reg_elements


os.chdir(open("../data_dir").read().strip())
# for filename in os.listdir("data/genomes/"):
#     if filename.endswith(".fa"):
#         print(filename)
#         print(len(cm.parse_genome("data/genomes/" + filename, chr1=True)["chr1"]))

# get_results("human", "data/hg19.cage_peak_phase1and2combined_coord.bed")
get_results("mouse", "data/cage/mm9.cage_peak_phase1and2combined_coord.bed")
get_results("chicken", "data/cage/galGal5.cage_peak_coord.bed")
get_results("monkey", "data/cage/rheMac8.cage_peak_coord.bed")
get_results("rat", "data/cage/rn6.cage_peak_coord.bed")
get_results("dog", "data/cage/canFam3.cage_peak_coord.bed")