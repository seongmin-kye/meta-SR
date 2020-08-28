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

In configure.py, specify the path to the directory. For example, in `meta-SR/configure.py` line 2:
```
save_path = '/root/home/voxceleb'
```
Then, run extract acoustic feature (mel filterbank-40).
```
python feat_extract/feature_extraction.py
```

#### Training examples
- Softmax:
```
python train.py --loss_type softmax --use_GC False --n_shot 1 --n_query 0 --use_variable False --nb_class_train 256
```
- Prototypical without global classification:
```
python train.py --loss_type prototypical --use_GC False --n_shot 1 --n_query 2 --use_variable True --nb_class_train 100
```
- Prototypical with global classification:
```
python train.py --loss_type prototypical --use_GC True --n_shot 1 --n_query 2 --use_variable True --nb_class_train 100
```
if you want to use fixed length query, set `--use_variable False`.

#### Evaluation
If you use __n-th__ folder __k-th__ checkpoint
- Speaker verification for full utterance:
```
python EER_full.py --n_folder n --cp_num k --data_type vox2
```
if you trained the model with VoxCeleb1, set `--data_type vox1`.

- Speaker verification for short utterance:
```

```
- Unseen speaker identification:
```

```

#### Pretrained models
A pretrained model can be downloaded from [here](https://drive.google.com/file/d/1uqRviTrmm578nw_OQgqtj3iAmc6eSnTI/view?usp=sharing).
Put this pretrained model to `meta-SR/saved_model/baseline_00n/`.
```
python EER_full.py --n_folder n --cp_num 100 --data_type vox2
```


