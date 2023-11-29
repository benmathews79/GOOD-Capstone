docker run --rm^
 --gpus all^
 -v c:/Users/benma/anaconda3/GOOD/dataset:/good/dataset:rw^
 -v c:/Users/benma/anaconda3/GOOD/good-main:/good:rw^
 -it --shm-size 8G goodimage:1.1
 

