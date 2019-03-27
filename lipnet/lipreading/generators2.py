import os

import keras
import numpy as np

from lipnet.lipreading.videos import Video

max_length = 10000


class ClassifyGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, datasets, align_hash, batch_size=32, curriculum=None, face_predictor_path=None,
                 vtype='face',
                 align_max_len=100,
                 dataset_path=None,
                 frames_n=None,
                 shuffle=True):
        self.batch_size = batch_size
        self.align_hash = align_hash
        self.list_IDs = list_IDs
        self.datasets = datasets
        self.shuffle = shuffle
        self.curriculum = curriculum
        self.vtype = vtype
        self.frames_n = frames_n
        self.align_max_len = align_max_len
        self.face_predictor_path = face_predictor_path
        self.dataset_path = dataset_path
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self, epoch=1, logs={}):
        self.indexes = np.arange(len(self.list_IDs))
        self.curriculum.update(epoch, self.shuffle)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_align(self, _id):
        return self.align_hash[_id.split('.')[0]]

    def __data_generation(self, list_IDs_temp):
        X_data = []
        Y_data = []
        label_length = []
        input_length = []
        source_str = []

        for i, ID in enumerate(list_IDs_temp):
            path = self.datasets[ID]
            video_id = os.path.splitext(path)[0].split('/')[-1]
            numpy_path = self.dataset_path + "/numpy/" + video_id + ".npy"
            video = Video(self.vtype, self.face_predictor_path, self.frames_n).from_numpy(numpy_path)
            # print("read video: " + str(i))
            align = self.get_align(path.split('/')[-1])
            video_unpadded_length = video.length
            if self.curriculum is not None:
                video, align, _ = self.curriculum.apply(video, align)
            X_data.append(video.data)
            Y_data.append(align.padded_label)
            label_length.append(align.label_length)  # CHANGED [A] -> A, CHECK!
            # input_length.append([video_unpadded_length - 2]) # 2 first frame discarded
            input_length.append(
                video.length)  # Just use the video padded length to avoid CTC No path found error (v_len < a_len)
            source_str.append(align.sentence)  # CHANGED [A] -> A, CHECK!

        source_str = np.array(source_str)
        label_length = np.array(label_length)
        input_length = np.array(input_length)
        Y_data = np.array(Y_data)
        X_data = np.array(X_data).astype(
            np.float32) / 255  # Normalize image data to [0,1], TODO: mean normalization over training data

        inputs = {'the_input': X_data,
                  'the_labels': Y_data,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([self.batch_size])}  # dummy data for dummy loss function
        return (inputs, outputs)
