from configure import save_path

# VoxCeleb2 train
TRAIN_AUDIO_VOX2 = save_path+'/voxceleb2/dev/wav'
TRAIN_FEAT_VOX2 = save_path+'/voxceleb2/dev/feat'

# VoxCeleb1 train
TRAIN_AUDIO_VOX1 = save_path+'/voxceleb1/dev/wav'
TRAIN_FEAT_VOX1 = save_path+'/voxceleb1/dev/feat'

# VoxCeleb1 test
TEST_AUDIO_VOX1 = save_path+'/voxceleb1/test/wav'
TEST_FEAT_VOX1= save_path+'/voxceleb1/test/feat'

USE_LOGSCALE = True
USE_NORM=  True
USE_DELTA = False
USE_SCALE = False

SAMPLE_RATE = 16000
FILTER_BANK = 40