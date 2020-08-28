# Meta-Learning for Short Utterance Speaker Recognition with Imbalance Length Pairs
Pytorch code for following paper:
* **Title** : Meta-Learning for Short Utterance Speaker Recognition with Imbalance Length Pairs. [[paper](https://arxiv.org/abs/2004.02863)]
* **Author** : Seong Min Kye, [Youngmoon Jung](https://github.com/jymsuper), [Hae Beom Lee](https://github.com/haebeom-lee), [Sung Ju Hwang](http://www.sungjuhwang.com), Hoirin Kim 
* **Conference** : Interspeech, 2020.

#### Data preparation

The following script can be used to download and prepare the VoxCeleb dataset for training. This preparation code is based on [VoxCeleb_trainer](https://github.com/clovaai/voxceleb_trainer), but slightly changed.

```
python dataprep.py --save_path /root/home/voxceleb --download --user USERNAME --password PASSWORD 
python dataprep.py --save_path /root/home/voxceleb --extract
python dataprep.py --save_path /root/home/voxceleb --convert
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.

#### Feature extraction

In configure.py, specify the path to the directory. For example, in meta-SR/configure.py line 2:
```
save_path = '/root/home/voxceleb'
```
