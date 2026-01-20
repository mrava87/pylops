# INSTALL

## 3.13
conda create -n pylops_313ft -c conda-forge python-free-threaded=3.13
conda install numpy scipy
pip install threadpoolctl
pip install -e .

## 3.14
conda create -n pylops_314ft -c conda-forge  python-freethreading=3.14
conda install numpy scipy
pip install threadpoolctl
pip install -e .

# RUN BASIC EXAMPLE
conda activate pylops
python3.13 example.py
python3.13 example.py --multithreaded # GIL multithreading
conda activate pylops_313ft
python3.13t example.py
python3.13t example.py --multithreaded # no-GIL multithreading
conda activate pylops_314ft
python3.14t example.py
python3.14t example.py --multithreaded # no-GIL multithreading

# RUN PYLOPS HSTACK
conda activate pylops
python3.13 pyhstack.py # GIL multithreading
conda activate pylops_313ft
OMP_NUM_THREADS=4; OPENBLAS_NUM_THREADS=4; MKL_NUM_THREADS=4; python3.13t -X gil=0 pyhstack.py # no-GIL multithreading
conda activate pylops_315ft
OMP_NUM_THREADS=4; OPENBLAS_NUM_THREADS=4; MKL_NUM_THREADS=4; python3.14t -X gil=0 pyhstack.py # no-GIL multithreading

# RUN PYLOPS BLOCKDIAG
conda activate pylops
python3.13 pyblockdiag.py # GIL multithreading
conda activate pylops_313ft
OMP_NUM_THREADS=4; OPENBLAS_NUM_THREADS=4; MKL_NUM_THREADS=4; python3.13t -X gil=0 pyblockdiag.py # no-GIL multithreading
conda activate pylops_314ft
OMP_NUM_THREADS=4; OPENBLAS_NUM_THREADS=4; MKL_NUM_THREADS=4; python3.14t -X gil=0 pyblockdiag.py # no-GIL multithreading
