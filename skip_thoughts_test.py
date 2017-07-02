from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("C:/Users/lenovo/Documents/TFmodels/models/skip_thoughts/skip_thoughts")



import numpy as np
import os.path
import scipy.spatial.distance as sd
from skip_thoughts.skip_thoughts import configuration
from skip_thoughts.skip_thoughts import encoder_manager


# Set paths to the model.
VOCAB_FILE = "C:/Users/lenovo/Documents/TFmodels/models/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02/skip_thoughts_uni_2017_02_02/vocab.txt"
EMBEDDING_MATRIX_FILE = "C:/Users/lenovo/Documents/TFmodels/models/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02\skip_thoughts_uni_2017_02_02/embeddings.npy"
CHECKPOINT_PATH = "C:/Users/lenovo/Documents/TFmodels/models/skip_thoughts/pretrained/skip_thoughts_uni_2017_02_02\skip_thoughts_uni_2017_02_02/model.ckpt-501424.data-00000-of-00001"
# The following directory should contain files rt-polarity.neg and
# rt-polarity.pos.
# MR_DATA_DIR = "/dir/containing/mr/data"


encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)