FROM centos:8

ENV PYTHON_VERSION "3.7.0"

RUN yum install -y libffi-devel gsl gsl-devel \
    && yum install -y gcc gcc-c++ make zlib-devel openssl-devel bzip2-devel git \
    && yum clean all

RUN curl -SLO https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar xvf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --prefix=/usr/local \
    && make \
    && make altinstall \
    && cd / \
    && rm -rf Python-${PYTHON_VERSION}*

RUN python3.7 -m pip install --upgrade pip &&  python3.7 -m pip install numpy Cython tqdm
RUN python3.7 -m pip install git+https://github.com/mattjj/pybasicbayes.git
RUN python3.7 -m pip install git+https://M0hSeN:batman8941607@gitlab.com/neuroscience-lab/pyhawkes.git
RUN python3.7 -m pip install git+https://M0hSeN:batman8941607@gitlab.com/neuroscience-lab/pyvlmc.git

CMD ["/bin/bash"]
