# Discovering Functional Connectivity 

Network analysis on population of Neurons to discover functional connectivity. 
This work is based on Hawkes process maintained in pyhawkes package.     

# Installation

## Python packages for build 

```bash
pip install numpy Cython
```
 
## Pyhawkes
 
```bash
git clone https://gitlab.com/neuroscience-lab/pyhawkes.git 
cd pyhawkes 
pip install -e .
```
## Requirement for static image save in plotly and bokeh

### From conda
```bash
conda install selenium phantomjs pillow plotly plotly-orca 
```

### From NodeJs 

```bash
npm install --unsafe-perm -g phantomjs-prebuilt electron@1.8.4 orca
```

### Instaling orca without an x11 server running 

* download latest orca.App image from https://github.com/plotly/orca/releases
* create a bash script:

 ```bash
#!/bin/bash
xvfb-run -a /root/Downloads/orca-1.2.1-x86_64.AppImage "$@"
```
* cp it in /bin
