# Use the official PyTorch base image as a starting point.
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Set the working directory.
WORKDIR /good

RUN apt-get update && apt-get install libgl1 -y
    
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq install libglib2.0-0

# Install additional dependencies.
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

# Copy your code to the container.
COPY . /good

# Install your code as an editable package.
RUN pip install -v -e .

RUN pip install numpy==1.24.4
RUN pip install yapf==0.40.1


# RUN python tools/train_good.py configs/good/phase1_depth.py

# CMD ["python","tools/train_good.py","configs/good/phase1_depth.py"]


