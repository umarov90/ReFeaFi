#!/bin/sh
import os
import subprocess

meme_id = {}
meme_tf = []

fimo_dir = "/media/user/30D4BACAD4BA9218/data_ubuntu/DeepRAG/data/fimo_jaspar/"
meme_dir = "/media/user/30D4BACAD4BA9218/data_ubuntu/DeepRAG/data/jaspar/"
for filename in os.listdir(meme_dir):
    if filename.endswith(".meme"):
        file = os.path.join(meme_dir, filename)
        with open(file) as f:
            for line in f:
                if line.startswith("MOTIF"):
                    vals = line.split(" ")
                    tf_name = vals[2].strip().upper()
                    meme_tf.append(tf_name)
                    meme_id[tf_name] = vals[1]
for tf in meme_tf:
    out_dir = fimo_dir + meme_id[tf]
    os.mkdir(out_dir)
    subprocess.run("/home/user/meme/bin/fimo --oc " + out_dir + "/prom --norc " + meme_dir + meme_id[tf] + ".meme /media/user/30D4BACAD4BA9218/data_ubuntu/DeepRAG/data/promoters.fa", shell=True)
    subprocess.run("/home/user/meme/bin/fimo --oc " + out_dir + "/enh --norc " + meme_dir + meme_id[tf] + ".meme /media/user/30D4BACAD4BA9218/data_ubuntu/DeepRAG/data/enhancers.fa", shell=True)
    with open(out_dir + "/info.txt", "w") as file:
        file.write(tf)

