import os

import numpy as np
import math

os.chdir("/home/user/data/DeepRAG/analysis")

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
    with open("preds/" + organism + "_chr1_PPde.txt") as file:
        for line in file:
            if(line.startswith("#")):
                continue
            vals = line.split("\t")
            if(float(vals[10]) > dt):
                p = (int(vals[2]) + int(vals[3])) / 2.0
                preds.append([p, float(vals[10])])
            mvs.append(float(vals[10]))

    with open("preds/" + organism + "_chr1_rev_PPde.txt") as file:
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
    with open("preds/" + organism + "_chr1.fa.gff3") as file:
        for line in file:
            vals = line.split("\t")
            p = int(vals[3]) + 200 
            preds.append([p, float(vals[5])])
            mvs.append(float(vals[5]))

    with open("preds/" + organism + "_chr1_rev.fa.gff3") as file:
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

def get_deepag(fpath, type):
    min_score = 100
    max_score = -100
    preds = []
    with open(fpath) as file:
        for line in file:
            vals = line.split("\t")
            score = float(vals[5])
            if vals[6] == ".":
                if type == "promoter":
                    continue
                score = score * 1.5
            try:
                score = math.log(score / (1 - score))
                if score < min_score:
                    min_score = score
                if score > max_score:
                    max_score = score
                if(vals[2] == "promoter/enhancer"):
                    pos = int(int(vals[3]) + (int(vals[4]) - int(vals[3])) / 2) - 1
                    preds.append([pos, score])
            except:
                pass
    preds.sort()
    return preds

def get_ef():
    preds = []
    all_scores = []
    with open("../data/enhancerfinder_step1_hg19.bed") as file:
        for line in file:
            if line.startswith("#"):
                continue
            vals = line.split("\t")
            if vals[0] != "chr1":
                continue
            scores = vals[3].split(";")
            score = 0.0
            for s in scores:
                score = score + float(s)
            score = score / len(scores)
            all_scores.append(score)
            chrp = int(int(vals[1]) + (int(vals[2]) - int(vals[1])) / 2) - 1
            #chrp = int(vals[1])
            preds.append([chrp, score])
    all_scores = np.asarray(all_scores)
    return preds


def get_results(organism):
    print(organism)
    deepag = get_deepag("../human_chr1_1iter.gff", type="promoter")
    deepag_noshift = get_deepag("../human_chr1_no_shift.gff", type="promoter")
    baseline = get_deepag("../human_baseline_chr1.gff", type="promoter")
    cage = get_cage(type="promoter")

    # ep3 = get_ep3(organism)
    # prompredict = get_prompredict(organism)


    # print("deeprag")
    # dtv = []
    # for dti in range(100):
    #     dt = -2.2 + dti*0.123
    #     res4 = compare(cage, deepag, dt)
    #     fpr4 = 1000000 * (res4[0]/(2*gl[organism]))
    #     dtv.append(str(res4[1]) + "," + str(fpr4))
    #
    # with open("dtv_deeprag_"+organism+".csv", 'w+') as f:
    #     for l in dtv:
    #         f.write(str(l) + "\n")
    #
    # print("deeprag_noshift")
    # dtv = []
    # for dti in range(100):
    #     dt = -2.2 + dti * 0.123
    #     res4 = compare(cage, deepag_noshift, dt)
    #     fpr4 = 1000000 * (res4[0] / (2 * gl[organism]))
    #     dtv.append(str(res4[1]) + "," + str(fpr4))
    #
    # with open("dtv_deeprag_noshift_" + organism + ".csv", 'w+') as f:
    #     for l in dtv:
    #         f.write(str(l) + "\n")
    #
    # print("baseline")
    # dtv = []
    # for dti in range(100):
    #     dt = -2.2 + dti*0.123
    #     res4 = compare(cage, baseline, dt)
    #     fpr4 = 1000000 * (res4[0]/(2*gl[organism]))
    #     dtv.append(str(res4[1]) + "," + str(fpr4))
    #
    # with open("dtv_baseline_"+organism+".csv", 'w+') as f:
    #     for l in dtv:
    #         f.write(str(l) + "\n")

    # print("ep3")
    # dtv = []
    # for dti in range(100):
    #     dt = -0.25 + dti * 0.011
    #     res1 = compare(cage, ep3, dt)
    #     fpr1 = 1000000 * (res1[0]/(2*gl[organism]))
    #     dtv.append(str(res1[1]) + "," + str(fpr1))
    #
    # with open("dtv_ep3_"+organism+".csv", 'w+') as f:
    #     for l in dtv:
    #         f.write(str(l) + "\n")
    #
    # print("prompredict")
    # dtv = []
    # for dti in range(100):
    #     dt = 2.8 + dti * 0.032
    #     res2 = compare(cage, prompredict, dt)
    #     fpr2 = 1000000 * (res2[0]/(2*gl[organism]))
    #     dtv.append(str(res2[1]) + "," + str(fpr2))
    #
    # with open("dtv_prompredict_"+organism+".csv", 'w+') as f:
    #     for l in dtv:
    #         f.write(str(l) + "\n")
    #
    # res1 = compare(cage, ep3, -0.18)
    # fpr1 = 1000000 * (res1[0]/(2*gl[organism]))
    #
    # res2 = compare(cage, prompredict, 2.8)
    # fpr2 = 1000000 * (res2[0]/(2*gl[organism]))
    #
    # res3 = compare(cage, deepag, 0.935)
    # fpr3 = 1000000 * (res3[0]/(2*gl[organism]))
    #
    # print(str(len(cage["chr1"])) + " " + str(fpr1) + " " + str(fpr2) + " " + str(fpr3) + " " + str(res1[4]) + " "+ str(res2[4]) + " "+ str(res3[4])+ " " + str(res1[1]) + " " + str(res2[1]) + " " + str(res3[1]) )
    #
    deepag = get_deepag("../human_chr1_no_shift.gff", type="enhancer")
    cage = get_cage(type="enhancer")
    ef = get_ef()

    print("deeprag")
    dtv = []
    for dti in range(100):
        dt = -2.2 + dti * 0.123
        res1 = compare(cage, deepag, dt)
        fpr1 = 1000000 * (res1[0] / (2 * gl[organism]))
        dtv.append(str(res1[1]) + "," + str(fpr1))

    with open("dtv_deeprag_" + organism + "_enhancer.csv", 'w+') as f:
        for l in dtv:
            f.write(str(l) + "\n")


    print("EnhancerFinder")
    dtv = []
    for dti in range(100):
        dt = -20 + dti * 0.4
        res3 = compare(cage, ef, dt)
        fpr3 = 1000000 * (res3[0] / (2 * gl[organism]))
        dtv.append(str(res3[1]) + "," + str(fpr3))

    with open("dtv_ef_" + organism + "_enhancer.csv", 'w+') as f:
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


def get_cage(type):
    reg_elements = {}
    if type == "promoter":
        with open("../data/hg19.cage_peak_phase1and2combined_coord.bed") as file:
            for line in file:
                vals = line.split("\t")
                chrn = vals[0]
                val = 1
                chrp = int(vals[7]) - 1
                reg_elements.setdefault(chrn,[]).append([chrp, val])
    else:
        with open("../data/human_permissive_enhancers_phase_1_and_2.bed") as file:
            for line in file:
                vals = line.split("\t")
                chrn = vals[0]
                val = 1
                chrp = int(vals[7]) - 1
                reg_elements.setdefault(chrn,[]).append([chrp, val])
    return reg_elements


os.chdir("/home/user/data/DeepRAG/analysis")
get_results("human")
#get_results("mouse", "mm9.cage_peak_phase1and2combined_coord.bed")
#get_results("chicken", "galGal5.cage_peak_coord.bed")
#get_results("monkey", "rheMac8.cage_peak_coord.bed")
#get_results("rat", "rn6.cage_peak_coord.bed")
#get_results("dog", "canFam3.cage_peak_coord.bed")