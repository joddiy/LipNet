import numpy as np


class Align(object):
    def __init__(self, absolute_max_string_len=32, label_func=None, align_map=None):
        self.label_func = label_func
        self.absolute_max_string_len = absolute_max_string_len
        self.align_map = align_map

    def from_file(self, path):
        """
        从文件中读取align
        """
        with open(path, 'r', encoding='utf-8-sig') as f:
            lines = f.read().splitlines()

        align = [(int(y[0]) / 1000, int(y[1]) / 1000, y[2]) for y in
                 [x.strip().split(" ") for x in lines if len(x.strip()) != 0]]
        self.build(align)
        return self

    def from_array(self, align):
        self.build(align)
        return self

    def build(self, align):
        self.align = self.strip(align, ['sp', 'sil'])  # align.align list
        self.sentence = self.get_sentence(align)  ## 空格隔开的 string
        self.label = self.get_label(self.sentence)
        self.padded_label = self.get_padded_label(self.label)

    def strip(self, align, items):
        return [sub for sub in align if sub[2] not in items]

    def get_sentence(self, align):
        return " ".join([y[-1] for y in align if y[-1] not in ['sp', 'sil']])

    def get_label(self, sentence):
        """
        根据传入的fuc处理label
        sentence: 空格隔开的 string
        """
        return self.label_func(sentence, self.align_map)

    def get_padded_label(self, label):
        """
        padding
        """
        padding = np.ones((self.absolute_max_string_len - len(label))) * -1
        return np.concatenate((np.array(label), padding), axis=0)

    @property
    def word_length(self):
        return len(self.sentence.split(" "))

    @property
    def sentence_length(self):
        return len(self.sentence)

    @property
    def label_length(self):
        return len(self.label)
