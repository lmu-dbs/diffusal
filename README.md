# DiffusAL
Diffusion-based Active Learning for Semi-Supervised Node Classification

# Getting Started
## install dependencies
``` python
python3 -m venv ./diffusal
source ./diffusal/bin/activate
pip install -U pip setuptools wheel
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install torch-geometric
pip install -r requirements.txt
```

## run experiments

``` 
(diffusal) python .\src\main.py
```

Expects that folder `<root>/data` exists and holds preprocessed data files. 
Creates `mlruns` folder in `<root>` and stores experiments there. 
Can be viewd in mlflow interface with ```mlflow server```.

