import numpy as np
import math

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

def get_deepag(organism):
    preds = []
    with open("../" + organism+"_chr1.gff") as file:
        for line in file:
            vals = line.split("\t")
            if(vals[2] == "promoter/enhancer"):
                pos = int(int(vals[3]) + (int(vals[4]) - int(vals[3])) / 2) - 1
                preds.append([pos, float(vals[5])])
    preds.sort()
    return preds

def get_results(organism, cage_file):
    #print(organism)
    ep3 = get_ep3(organism)
    prompredict = get_prompredict(organism)
    deepag = get_deepag(organism)
    cage = get_cage(cage_file)    
    # print("ep3")
    # dtv = []
    # for dti in range(100):
    #     dt = -0.25 + dti * 0.011
    #     res1 = compare(cage, ep3, dt)
    #     fpr1 = 1000000 * (res1[0]/(2*gl[organism]))
    #     dtv.append(str(res1[1]) + "," + str(fpr1))

    # with open("dtv_ep3_"+organism+".csv", 'w+') as f:
    #     for l in dtv:
    #         f.write(str(l) + "\n")
    #print(str(fp) + " " + str(fpr) + " " + str(recall) + " " + str(precision) + " " + str(f1))
    

    # print("prompredict")
    # dtv = []
    # for dti in range(100):
    #     dt = 2.8 + dti * 0.032
    #     res2 = compare(cage, prompredict, dt)
    #     fpr2 = 1000000 * (res2[0]/(2*gl[organism]))
    #     dtv.append(str(res2[1]) + "," + str(fpr2))
    #     #print(str(fp) + " " + str(fpr) + " " + str(recall) + " " + str(precision) + " " + str(f1))

    # with open("dtv_prompredict_"+organism+".csv", 'w+') as f:
    #     for l in dtv:
    #         f.write(str(l) + "\n")

    # print("deeprag")
    # dtv = []
    # dt = 0.05
    # while(dt < 1):        
    #     res4 = compare(cage, deepag, dt)
    #     fpr4 = 1000000 * (res4[0]/(2*gl[organism]))
    #     dtv.append(str(res4[1]) + "," + str(fpr4))
    #     #print(dt)
    #     if(dt < 0.9):
    #         dt = dt + 0.005
    #     elif(dt < 0.99):
    #         dt = dt + 0.0005
    #     else:
    #         dt = dt + 0.00005


    # with open("dtv_deeprag_"+organism+".csv", 'w+') as f:
    #     for l in dtv:
    #         f.write(str(l) + "\n")

    res1 = compare(cage, ep3, -0.18)
    fpr1 = 1000000 * (res1[0]/(2*gl[organism]))
    #print(str(fp) + " " + str(fpr) + " " + str(recall) + " " + str(precision) + " " + str(f1))
    

    #print("prompredict")
    res2 = compare(cage, prompredict, 2.8)
    fpr2 = 1000000 * (res2[0]/(2*gl[organism]))
    #print(str(fp) + " " + str(fpr) + " " + str(recall) + " " + str(precision) + " " + str(f1))

    #print("promid")
    res3 = compare(cage, deepag, 0.935)
    fpr3 = 1000000 * (res3[0]/(2*gl[organism]))
    #print(str(res1[1]) + " " + str(res2[1]) + " " + str(res3[1]) + " " + str(res1[2]) + " " + str(res2[2]) + " " + str(res3[2]) + " " + str(res1[3]) + " " + str(res2[3]) + " " + str(res3[3]))
    print(str(len(cage["chr1"])) + " " + str(fpr1) + " " + str(fpr2) + " " + str(fpr3) + " " + str(res1[4]) + " "+ str(res2[4]) + " "+ str(res3[4])+ " " + str(res1[1]) + " " + str(res2[1]) + " " + str(res3[1]) )
    

    #print(str(res1[1]) + " " + str(res2[1]) + " " + str(res3[1]) + " " + str(res1[2]) + " " + str(res2[2]) + " " + str(res3[2]) + " " + str(res1[3]) + " " + str(res2[3]) + " " + str(res3[3]))
    #print(str(len(cage["chr1+"]) + len(cage["chr1-"]))  + " " + str(fpr1) + " " + str(fpr2) + " " + str(fpr3) + " " + str(res1[1]) + " " + str(res2[1]) + " " + str(res3[1]) )
    #print(str(fp) + " " + str(fpr) + " " + str(recall) + " " + str(precision) + " " + str(f1))

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
    return fp, recall, precision, f1, fp / tp

def get_cage(path):
    reg_elements = {}
    with open("../cage.bed") as file:
        for line in file:
            vals = line.split("\t")
            chrn = vals[0]
            val = 1
            chrp = int(vals[7]) - 1
            reg_elements.setdefault(chrn,[]).append([chrp, val])
    return reg_elements


get_results("human", "../cage.bed")
#get_results("mouse", "mm9.cage_peak_phase1and2combined_coord.bed")
#get_results("chicken", "galGal5.cage_peak_coord.bed")
#get_results("monkey", "rheMac8.cage_peak_coord.bed")
#get_results("rat", "rn6.cage_peak_coord.bed")
#get_results("dog", "canFam3.cage_peak_coord.bed")