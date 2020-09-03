# Meta-Learning for Short Utterance Speaker Recognition with Imbalance Length Pairs
Pytorch code for following paper:
* **Title** : Meta-Learning for Short Utterance Speaker Recognition with Imbalance Length Pairs. [[paper](https://arxiv.org/abs/2004.02863)]
* **Author** : Seong Min Kye, [Youngmoon Jung](https://github.com/jymsuper), [Hae Beom Lee](https://haebeom-lee.github.io/), [Sung Ju Hwang](http://www.sungjuhwang.com), Hoirin Kim 
* **Conference** : Interspeech, 2020.

### Abstract
<img align="middle" width="1000" src="https://github.com/seongmin-kye/meta-SR/blob/master/overview.png">

In practical settings, a speaker recognition system needs to identify a speaker given a short utterance, while the enrollment utterance may be relatively long. However, existing speaker recognition models perform poorly with such short utterances. To solve this problem, we introduce a meta-learning framework for imbalance length pairs. Specifically, we use a Prototypical Networks and train it with a support set of long utterances and a query set of short utterances of varying lengths. Further, since optimizing only for the classes in the given episode may be insufficient for learning discriminative embeddings for unseen classes, we additionally enforce the model to classify both the support and the query set against the entire set of classes in the training set. By combining these two learning schemes, our model outperforms existing state-of-the-art speaker verification models learned with a standard supervised learning framework on short utterance (1-2 seconds) on the VoxCeleb datasets. We also validate our proposed model for unseen speaker identification, on which it also achieves significant performance gains over the existing approaches.

### Requirements
* Python 3.6
* Pytorch 1.3.1

### Data preparation

The following script can be used to download and prepare the VoxCeleb dataset for training. This preparation code is based on [**VoxCeleb_trainer**](https://github.com/clovaai/voxceleb_trainer), but slightly changed.

```
python dataprep.py --save_path /root/home/voxceleb --download --user USERNAME --password PASSWORD 
python dataprep.py --save_path /root/home/voxceleb --extract
python dataprep.py --save_path /root/home/voxceleb --convert
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.

### Feature extraction

In configure.py, specify the path to the directory. For example, in `meta-SR/configure.py` line 2:
```
save_path = '/root/home/voxceleb'
```
Then, extract acoustic feature (mel filterbank-40).
```
python feat_extract/feature_extraction.py
```

### Training examples
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

### Evaluation
If you use __n-th__ folder & __k-th__ checkpoint
- Speaker verification for full utterance:
```
python EER_full.py --n_folder n --cp_num k --data_type vox2
```
if you trained the model with VoxCeleb1, set `--data_type vox1`.

- Speaker verification for short utterance:
```
python EER_short.py --n_folder n --cp_num k --test_length 100
```
ex) test on 2-second utterance, set `--test_length 200`.

- Unseen speaker identification:
```
python identification.py --n_folder n --cp_num k --nb_class_test 100 --test_length 100
```

### Pretrained model
A pretrained model can be downloaded from [here](https://drive.google.com/file/d/1uqRviTrmm578nw_OQgqtj3iAmc6eSnTI/view?usp=sharing). 
If you put this model into baseline_000, and run following scrpt, you can get `EER 2.08`.
```
python EER_full.py --n_folder 0 --cp_num 100 --data_type vox2
```

### Citation
Please cite the following if you make use of the code.
```
@inproceedings{kye2020meta,
  title={Meta-Learning for Short Utterance Speaker Recognition with Imbalance Length Pairs},
  author={Kye, Seong Min and Jung, Youngmoon and Lee, Hae Beom and Hwang, Sung Ju and Kim, Hoirin},
  booktitle={Interspeech},
  year={2020}
}
```
### Acknowledgments
This code is based on the implementation of [**SR_tutorial**](https://github.com/jymsuper/SpeakerRecognition_tutorial) and [**VoxCeleb_trainer**](https://github.com/clovaai/voxceleb_trainer). I would like to thank Youngmoon Jung, Joon Son Chung and Sung Ju Hwang for helpful discussions.
