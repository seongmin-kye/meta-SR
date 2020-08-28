# Meta-Learning for Short Utterance Speaker Recognition with Imbalance Length Pairs
Pytorch implementation of "Meta-Learning for Short Utterance Speaker Recognition with Imbalance Length Pairs"


#### Data preparation

The following script can be used to download and prepare the VoxCeleb dataset for training. These codes are based on VoxCeleb_trainer

```
python dataprep.py --save_path /root/home/voxceleb --download --user USERNAME --password PASSWORD 
python dataprep.py --save_path /root/home/voxceleb --extract
python dataprep.py --save_path /root/home/voxceleb --convert
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.
