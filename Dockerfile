FROM nvidia/cuda:11.3.1-cudnn8-devel-centos8

ENV HOME /home/pywick

RUN yum install -y epel-release && yum install -y dnf-plugins-core && yum config-manager --set-enabled powertools
RUN yum update -y && yum -y install atop bzip2-devel ca-certificates cmake curl git grep htop less libffi-devel hdf5-devel libjpeg-devel xz-devel libuuid-devel libXext libSM libXrender make nano openssl-devel sed screen tini vim wget unzip

RUN yum groupinstall -y "Development Tools"

RUN wget https://www.python.org/ftp/python/3.9.5/Python-3.9.5.tgz
RUN tar xvf Python-3.9.5.tgz && cd Python-3.9*/ && ./configure --enable-optimizations && make altinstall && cd .. && rm -rf Python*

RUN cd /usr/bin && rm python3 pip3 && ln -s /usr/local/bin/python3.9 python && ln -s /usr/local/bin/python3.9 python3 && ln -s /usr/local/bin/pip3.9 pip3 && ln -s /usr/local/bin/pip3.9 pip
RUN pip install --upgrade pip setuptools wheel

### Pytorch V1.8.2 + CUDA (py3.9_cuda11.1_cudnn7.6.3_0)
RUN pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

## MacOS currently not supported for CUDA or LTS
#RUN pip install torch torchvision torchaudio

RUN mkdir -p /home && rm -rf $HOME
RUN cd /home && git clone https://github.com/achaiah/pywick
# To build from a different branch or tag specify per example below
#RUN cd $HOME && git checkout WIP2

# install requirements
RUN pip install versioned-hdf5
RUN pip install --upgrade -r $HOME/requirements.txt

ENV PYTHONPATH=/home:$HOME:$HOME/configs
WORKDIR $HOME

RUN chmod -R +x $HOME/*.sh

CMD ["/bin/bash", "/home/pywick/entrypoint.sh"]

###########
# Build with:
#   git clone https://github.com/achaiah/pywick
#   cd pywick
#   docker build -t "achaiah/pywick:latest" .
#
# Run 17flowers demo with:
#   docker run --rm -it --ipc=host --init -e demo=true achaiah/pywick:latest
# Optionally specify local dir where you want to save output: docker run --rm -it --ipc=host -v your_local_out_dir:/jobs/17flowers --init -e demo=true achaiah/pywick:latest
# Run container that just stays up (for your own processes):
#   docker run --rm -it --ipc=host -v <your_local_data_dir>:<container_data_dir> -v <your_local_out_dir>:<container_out_dir> --init achaiah/pywick:latest
###########