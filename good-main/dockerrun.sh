docker run --rm \
 --gpus all \
 -e DISPLAY=$DISPLAY \
 -v /home/adam.hawkins.net/workspace/vt/GOOD-Capstone/good-main:/good:rw \
 -it --shm-size 8G goodimage
 

