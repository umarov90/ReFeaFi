# DeepREFind: Genome-wide prediction of regulatory elements driving transcription initiation
![Framework](framework.png)
## Installation

Simply clone this repository and run predict.py to use the pre-trained models. 
DeepREFind requires ```tensorflow==1.7.0```:
```sh
pip install tensorflow==1.7.0
```
OR
```sh
pip install tensorflow-gpu==1.7.0
```
for the GPU version. In this case you also need to install CUDA9 and cuDNN7. 
## Usage
DeepREFind can be run from the command line:
```sh
python predict.py -I hg19.fa -O human_regulatory_regions.gff
```
Required parameters:
 - ```-I```: Input fasta file.
 - ```-O```: Output gff file.

Optional parameters:
 - ```-D```: Minimum soft distance between the predicted TSS, defaults to 1000.
 - ```-C```: Comma separated list of chromosomes to use for promoter prediction, defaults to all.
 - ```-T```: Decision threshold for the prediction model, defaults to 0.5.
 
To calculate dependency score used for the pair maps, run dependency_score.py:
```sh
python dependency_score.py promoters.fa 495:505 460:475 
```
Where first parameter is file with sequences and the next two represent regions of interest. 