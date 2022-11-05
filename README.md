
# HF-FSL 


dataset: CUB-200-2011, which can download from https://data.caltech.edu/records/65de6-vp158


## pretrain
'''bash
python pretrain.py --checkpoint=./pretrain -lambda1 0.0001 -m1 0.1
'''


## imprint+ft 
python imprint_ft.py --checkpoint=./ft -lambda1 0.0001 -m1 0.1 -lambda2 0.0001 -m2 0.1