This benchmark runs in image pytorch/pytorch:latest
it worked on Ubuntu 16.04 Driver 460.56 using image pytorch/pytorch:latest  *ver 1.8.0*

bash -c 'apt -y update;apt -y upgrade; apt install -y python3; apt install -y python3-pip; apt install -y git;git clone https://github.com/jjziets/pytorch-benchmark-volta.git; cd pytorch-benchmark-volta; pip install -r requirement.txt;chmod +x mytest.sh;./mytest.sh'

another way is to spin up a docker container and run the benshmark in there

sudo docker run --gpus all -it --shm-size=1g --ulimit memlock=-1 --rm pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime  /bin/bash
