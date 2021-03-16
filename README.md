This benchmark runs in image

image pytorch/pytorch:latest
it worked on Ubuntu 16.04 Driver 460.56 using image pytorch/pytorch:latest 1.8.0

bash -c 'apt -y update;apt -y upgrade; apt install -y python3; apt install -y python3-pip; apt install -y git;git clone https://github.com/jjziets/pytorch-benchmark-volta.git; cd pytorch-gpu-benchmark; pip install -r requirement.txt;./mytest.sh'
