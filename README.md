This benchmark runs in image pytorch/pytorch:latest
I suggest running it on the host machine in a docker container as below

sudo docker run --gpus all -it --shm-size=5g --ulimit memlock=-1 --rm pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime  /bin/bash

bash -c 'apt -y update;apt -y upgrade; apt install -y python3; apt install -y python3-pip; apt install -y git;git clone https://github.com/jjziets/pytorch-benchmark-volta.git; cd pytorch-benchmark-volta; pip install -r requirement.txt;chmod +x mytest.sh;./mytest.sh'
