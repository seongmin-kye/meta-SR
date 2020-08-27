# Feature path
TRAIN_FEAT_DIR_2 = '/media/user/samsung_1TB/DB/voxceleb2/vox2_dev_wav/feat/train_logfbank_nfilt40_scipy_hamming' #train_Vox2
TRAIN_FEAT_DIR_1 = '/media/user/samsung_1TB/DB/voxceleb1/vox1_dev_wav/feat/train_logfbank_nfilt40_scipy_hamming' #train_Vox1
TEST_FEAT_DIR = '/media/user/samsung_1TB/DB/voxceleb1/vox1_test_wav/feat/test_logfbank_nfilt40_scipy_hamming'    #test_Vox1

# trial pair set txt file
VERI_TEST_DIR = '/media/user/samsung_1TB/DB/voxceleb1/trial_pair_Verification.txt' #Original Vox1 test set

# Training context window size
NUM_WIN_SIZE = 200 #200ms == 2 seconds
SHORT_SIZE = 100   #100ms == 1 seconds

# Settings for feature extraction
USE_NORM = True
USE_SCALE = False
SAMPLE_RATE = 16000