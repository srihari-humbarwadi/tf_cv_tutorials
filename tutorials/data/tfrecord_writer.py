import json
import os

import numpy as np

import tensorflow as tf

print('TensorFlow:', tf.__version__)
np.random.seed(25)


class TFrecordWriter:

    def __init__(self, n_samples, n_shards, output_dir='', prefix=''):
        self.n_samples = n_samples
        self.n_shards = n_shards
        self._step_size = self.n_samples // self.n_shards + 1
        self.prefix = prefix
        self.output_dir = output_dir
        self._buffer = []
        self._file_count = 1

    def _make_example(self, image, boxes, classes):
        feature = {
            'image':
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'xmins':
                tf.train.Feature(float_list=tf.train.FloatList(value=boxes[:, 0])),
            'ymins':
                tf.train.Feature(float_list=tf.train.FloatList(value=boxes[:, 1])),
            'xmaxs':
                tf.train.Feature(float_list=tf.train.FloatList(value=boxes[:, 2])),
            'ymaxs':
                tf.train.Feature(float_list=tf.train.FloatList(value=boxes[:, 3])),
            'classes':
                tf.train.Feature(int64_list=tf.train.Int64List(value=classes))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _write_tfrecord(self, tfrecord_path):
        print('writing {} samples in {}'.format(len(self._buffer),
                                                tfrecord_path))
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for (image, boxes, classes) in self._buffer:
                example = self._make_example(image, boxes, classes)
                writer.write(example.SerializeToString())

    def push(self, image, boxes, classes):
        self._buffer.append([image, boxes, classes])
        if len(self._buffer) == self._step_size:
            fname = self.prefix + '_000' + str(self._file_count) + '.tfrecord'
            tfrecord_path = os.path.join(self.output_dir, fname)
            self._write_tfrecord(tfrecord_path)
            self._clear_buffer()
            self._file_count += 1

    def flush_last(self):
        if len(self._buffer):
            fname = self.prefix + '_000' + str(self._file_count) + '.tfrecord'
            tfrecord_path = os.path.join(self.output_dir, fname)
            self._write_tfrecord(tfrecord_path)

    def _clear_buffer(self):
        self._buffer = []


if __name__ == '__main__':
    shapes_dataset_dir = '../tutorials/data/shapes_dataset/'
    class_map = {'circle': 0, 'rectangle': 1}
    n_shards = 4
    tf_record_dir = '../tutorials/data/shapes_dataset_tfrecords'

    with open(shapes_dataset_dir + 'dataset.json', 'r') as fp:
        dataset_json = json.load(fp)

    all_image_names = list(dataset_json.keys())
    print('Found {} images'.format(len(all_image_names)))

    indices = np.arange(len(all_image_names))
    np.random.shuffle(indices)

    train_image_names = all_image_names[:2500]
    val_image_names = all_image_names[2500:]

    print('Splitting dataset into {} training images and {} validation images'.format(len(train_image_names),
                                                                                      len(val_image_names)))

    train_tf_record_writer = TFrecordWriter(n_samples=len(train_image_names),
                                            n_shards=n_shards,
                                            output_dir=tf_record_dir,
                                            prefix='train')

    for image_name in train_image_names:
        boxes = []
        classes = []

        with tf.io.gfile.GFile(shapes_dataset_dir + 'images/' + image_name, 'rb') as fp:
            image = fp.read()

        for obj in dataset_json[image_name]:
            boxes.append(obj['box'])
            classes.append(class_map[obj['category']])
        train_tf_record_writer.push(image, np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32))
    train_tf_record_writer.flush_last()

    val_tf_record_writer = TFrecordWriter(n_samples=len(val_image_names),
                                          n_shards=n_shards,
                                          output_dir=tf_record_dir,
                                          prefix='val')

    for image_name in val_image_names:
        boxes = []
        classes = []

        with tf.io.gfile.GFile(shapes_dataset_dir + 'images/' + image_name, 'rb') as fp:
            image = fp.read()

        for obj in dataset_json[image_name]:
            boxes.append(obj['box'])
            classes.append(class_map[obj['category']])
        val_tf_record_writer.push(image, np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32))
    val_tf_record_writer.flush_last()
