import random
import math
import tqdm
from datetime import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import efficientnet.tfkeras as efn
from sklearn.model_selection import train_test_split
import imgaug

LOG_DIR = "logs/"
image_size = 100
batch_size = 16

test_files = tf.io.gfile.glob(
    "/mnt/aitrics_ext/ext01/mandy/data/imagenet-5-categories/test/*.jpg"
)
train_files = tf.io.gfile.glob(
    "/mnt/aitrics_ext/ext01/mandy/data/imagenet-5-categories/train/*.jpg"
)
train_files, sub_train_files = train_test_split(train_files, test_size=150)


def random_brightness(image):
    return tf.image.random_brightness(image, 0.4)


def random_contrast(image):
    return tf.image.random_contrast(image, 0.3, 1.7)


def random_saturation(image):
    return tf.image.random_saturation(image, 0, 3)


def random_flip_left_right(image):
    return tf.image.random_flip_left_right(image)


def random_crop(image):
    a, b = tf.random.uniform((2,), image_size // 2, image_size, dtype=tf.int32)
    image = tf.image.random_crop(image, [a, b, 3])
    return tf.image.resize(image, (image_size, image_size))


def random_hue(image):
    return tf.image.random_hue(image, 0.05)


augs_color = [random_brightness, random_contrast]
augs_str = [random_flip_left_right, random_crop]
only_one_of = [random_saturation, random_hue]


def aug(image):
    augs = augs_color + augs_str + [random.choice(only_one_of)]
    random.shuffle(augs)
    for i in augs:
        image = i(image)
    return image


class DataGeneratorSiam(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        pics = []
        names = self.data[index * self.batch_size : (index + 1) * self.batch_size]
        for i in names:
            pics.append(
                tf.image.resize(
                    tf.io.decode_jpeg(tf.io.read_file(i)), (image_size, image_size)
                )
            )
        images = tf.stack(pics, 0) / 255.0
        return tf.map_fn(aug, images), tf.map_fn(aug, images)

    def on_epoch_end(self):
        np.random.shuffle(self.data)


class DataGenerator(DataGeneratorSiam):
    def __init__(self, data, batch_size, isaug=False):
        self.data = data
        self.batch_size = batch_size
        self.isaug = isaug
        self.on_epoch_end()
        self.label_dict = {"airplane": 0, "car": 1, "cat": 2, "dog": 3, "elephant": 4}

    def __getitem__(self, index):
        pics = []
        names = self.data[index * self.batch_size : (index + 1) * self.batch_size]
        labels = [i.split("/")[-1].split("_")[0] for i in names]
        for i in names:
            pics.append(
                tf.image.resize(
                    tf.io.decode_jpeg(tf.io.read_file(i)), (image_size, image_size)
                )
            )
        images = tf.stack(pics, 0) / 255.0
        if self.isaug:
            images = tf.map_fn(aug, images)
        one_hot = tf.one_hot([self.label_dict[i] for i in labels], len(self.label_dict))
        return images, one_hot


# for x, y in DataGenerator(test_files, batch_size, False):
#     break
#
# N = math.ceil(math.sqrt(batch_size))
# plt.imshow(imgaug.draw_grid(np.array(x.numpy() * 255, dtype="uint8"), cols=N, rows=N))
# plt.show()

# for i1, i2 in DataGeneratorSiam(train_files, batch_size):
#     break
#
# N = math.ceil(math.sqrt(batch_size))
# plt.imshow(imgaug.draw_grid(np.array(i1.numpy() * 255, dtype="uint8"), cols=N, rows=N))
# plt.show()
# plt.imshow(imgaug.draw_grid(np.array(i2.numpy() * 255, dtype="uint8"), cols=N, rows=N))
# plt.show()

# Baseline
inputs = tf.keras.Input(shape=(image_size,image_size,3))
x = efn.EfficientNetB0(include_top=False, weights='imagenet')(inputs)
vec = tf.keras.layers.GlobalMaxPool2D()(x)

di = tf.keras.layers.Dense(128,'sigmoid')(vec)
cls_out = tf.keras.layers.Dense(5, activation='softmax')(di)

cls_model = tf.keras.Model(inputs, cls_out)

logdir = "logs/baseline/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False)

cls_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.00001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=tf.keras.metrics.CategoricalAccuracy())

history = cls_model.fit(
    DataGenerator(sub_train_files, batch_size, True),
    epochs=200,
    validation_data = DataGenerator(test_files, batch_size, False),
    callbacks=[tensorboard_callback])

if __name__ == "__main__":
    pass
