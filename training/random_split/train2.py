import sys
import os

sys.path.append(os.getcwd())

from lipnet.lipreading.generators2 import ClassifyGenerator

from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from lipnet.lipreading.generators import RandomSplitGenerator
from lipnet.lipreading.callbacks import Statistics, Visualize
from lipnet.lipreading.curriculums import Curriculum
from lipnet.core.decoders import Decoder
from lipnet.lipreading.helpers import labels_to_text
from lipnet.utils.spell import Spell
from lipnet.model2 import LipNet
import numpy as np
import datetime

np.random.seed(55)

CURRENT_PATH = os.path.join(os.getcwd(), 'assets')
DATASET_DIR = os.path.join(CURRENT_PATH, 'datasets2')
OUTPUT_DIR = os.path.join(CURRENT_PATH, 'results')
LOG_DIR = os.path.join(CURRENT_PATH, 'logs')

FACE_PREDICTOR = os.path.join(CURRENT_PATH, '..', 'evaluation', 'models', 'shape_predictor_68_face_landmarks.dat')
PREDICT_GREEDY = False
PREDICT_BEAM_WIDTH = 200
PREDICT_DICTIONARY = os.path.join(CURRENT_PATH, '..', 'common', 'dictionaries', 'grid.txt')


def curriculum_rules(epoch):
    return {'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0.05}


def train(run_name, max_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size):
    curriculum = Curriculum(curriculum_rules)

    lip_gen = RandomSplitGenerator(dataset_path=DATASET_DIR,
                                   minibatch_size=minibatch_size,
                                   img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                                   absolute_max_string_len=absolute_max_string_len,
                                   curriculum=curriculum).build(val_split=0.2)

    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=lip_gen.get_output_size())
    lipnet.summary()

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)

    training_generator = ClassifyGenerator(range(len(lip_gen.train_list)), lip_gen.train_list, lip_gen.align_hash,
                                           minibatch_size,
                                           curriculum=curriculum,
                                           face_predictor_path=FACE_PREDICTOR,
                                           align_max_len=absolute_max_string_len,
                                           vtype='face',
                                           frames_n=frames_n)
    validation_generator = ClassifyGenerator(range(len(lip_gen.val_list)), lip_gen.val_list, lip_gen.align_hash,
                                             minibatch_size,
                                             curriculum=curriculum,
                                             face_predictor_path=FACE_PREDICTOR,
                                             align_max_len=absolute_max_string_len,
                                             vtype='face',
                                             frames_n=frames_n,
                                             shuffle=False)

    lipnet.model.fit_generator(generator=training_generator,
                               validation_data=validation_generator,
                               use_multiprocessing=True,
                               epochs=max_epoch,
                               workers=2,
                               # callbacks=[checkpoint, statistics, visualize, lip_gen, tensorboard, csv_logger],
                               )


if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    train(run_name, 20, 3, 100, 50, 100, 32, 10)
