# Installation
```bash
conda create -y -n evreal python=3.8
conda activate evreal
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

# Download dataset 

```bash
pip install -r tools/requirements.txt
./tools/download_ECD.sh
python tools/bag_to_npy.py data/ECD --remove
```

# Evaluation
```bash
python eval.py -m m1 m2 -c std -d ECD -qm mse ssim lpips
```

