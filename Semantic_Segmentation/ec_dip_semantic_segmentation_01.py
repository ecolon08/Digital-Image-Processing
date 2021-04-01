# import libraries
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# https://www.tensorflow.org/tutorials/images/segmentation
# load oxford pet dataset
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


def normalize(input_image, input_mask):

    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1

    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


# dataset already contains the train, validate, and test splits
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 16
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
test = dataset['test'].map(load_image_test)

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


for image, mask in train.take(1):
    sample_image, sample_mask = image, mask

display([sample_image, sample_mask])

# Define the model

inputs = tf.keras.layers.Input(shape=(128, 128, 3))

# DOWNSAMPLING

conv_block_1 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
conv_block_1 = tf.keras.layers.BatchNormalization()(conv_block_1)
conv_block_1 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same')(conv_block_1)

conv_block_1 = tf.keras.layers.BatchNormalization()(conv_block_1)
conv_block_1 = tf.keras.layers.Activation('relu')(conv_block_1)
conv_pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_block_1)

# block 2
conv_block_2 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same')(conv_pool_1)
conv_block_2 = tf.keras.layers.BatchNormalization()(conv_block_2)
conv_block_2 = tf.keras.layers.Activation('relu')(conv_block_2)

conv_block_2 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same')(conv_block_2)
conv_block_2 = tf.keras.layers.BatchNormalization()(conv_block_2)
conv_block_2 = tf.keras.layers.Activation('relu')(conv_block_2)
conv_pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_block_2)

# block 3
conv_block_3 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same')(conv_pool_2)
conv_block_3 = tf.keras.layers.BatchNormalization()(conv_block_3)
conv_block_3 = tf.keras.layers.Activation('relu')(conv_block_3)

conv_block_3 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same')(conv_block_3)
conv_block_3 = tf.keras.layers.BatchNormalization()(conv_block_3)
conv_block_3 = tf.keras.layers.Activation('relu')(conv_block_3)
conv_pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_block_3)

# block 4
conv_block_4 = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(conv_pool_3)
conv_block_4 = tf.keras.layers.BatchNormalization()(conv_block_4)
conv_block_4 = tf.keras.layers.Activation('relu')(conv_block_4)

conv_block_4 = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(conv_block_4)
conv_block_4 = tf.keras.layers.BatchNormalization()(conv_block_4)
conv_block_4 = tf.keras.layers.Activation('relu')(conv_block_4)
conv_pool_4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv_block_4)

# block 5
conv_block_5 = tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), padding='same')(conv_pool_4)
conv_block_5 = tf.keras.layers.BatchNormalization()(conv_block_5)
conv_block_5 = tf.keras.layers.Activation('relu')(conv_block_5)

conv_block_5 = tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), padding='same')(conv_block_5)
conv_block_5 = tf.keras.layers.BatchNormalization()(conv_block_5)
conv_block_5 = tf.keras.layers.Activation('relu')(conv_block_5)

conv_block_5 = tf.keras.layers.Dropout(0.5)(conv_block_5)

# UPSAMPLING

# block 6
up_conv_6 = tf.keras.layers.Concatenate()([conv_block_4, tf.keras.layers.Conv2DTranspose(512, kernel_size=(2, 2), strides=2)(conv_block_5)])
conv_block_6 = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(up_conv_6)
conv_block_6 = tf.keras.layers.BatchNormalization()(up_conv_6)
conv_block_6 = tf.keras.layers.Activation('relu')(conv_block_6)
conv_block_6 = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), padding='same')(conv_block_6)
conv_block_6 = tf.keras.layers.BatchNormalization()(up_conv_6)
conv_block_6 = tf.keras.layers.Activation('relu')(conv_block_6)

conv_block_6 = tf.keras.layers.Dropout(0.5)(conv_block_6)

# block 7
up_conv_7 = tf.keras.layers.Concatenate()([conv_block_3, tf.keras.layers.Conv2DTranspose(256, kernel_size=(2, 2), strides=2)(conv_block_6)])
conv_block_7 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same')(up_conv_7)
conv_block_7 = tf.keras.layers.BatchNormalization()(up_conv_7)
conv_block_7 = tf.keras.layers.Activation('relu')(conv_block_7)
conv_block_7 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same')(conv_block_7)
conv_block_7 = tf.keras.layers.BatchNormalization()(up_conv_7)
conv_block_7 = tf.keras.layers.Activation('relu')(conv_block_7)

conv_block_7 = tf.keras.layers.Dropout(0.5)(conv_block_7)

# block 8
up_conv_8 = tf.keras.layers.Concatenate()([conv_block_2, tf.keras.layers.Conv2DTranspose(128, kernel_size=(2, 2), strides=2)(conv_block_7)])
conv_block_8 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same')(up_conv_8)
conv_block_8 = tf.keras.layers.BatchNormalization()(up_conv_8)
conv_block_8 = tf.keras.layers.Activation('relu')(conv_block_8)
conv_block_8 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same')(conv_block_8)
conv_block_8 = tf.keras.layers.BatchNormalization()(up_conv_8)
conv_block_8 = tf.keras.layers.Activation('relu')(conv_block_8)

conv_block_8 = tf.keras.layers.Dropout(0.5)(conv_block_8)

# block 9
up_conv_9 = tf.keras.layers.Concatenate()([conv_block_1, tf.keras.layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=2)(conv_block_8)])
conv_block_9 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same')(up_conv_9)
conv_block_9 = tf.keras.layers.BatchNormalization()(up_conv_9)
conv_block_9 = tf.keras.layers.Activation('relu')(conv_block_9)
conv_block_9 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same')(conv_block_9)
conv_block_9 = tf.keras.layers.BatchNormalization()(up_conv_9)
conv_block_9 = tf.keras.layers.Activation('relu')(conv_block_9)

conv_block_9 = tf.keras.layers.Dropout(0.5)(conv_block_9)

# output layer
last = tf.keras.layers.Conv2DTranspose(3, 2, strides=1, padding='same')(conv_block_9)

model = tf.keras.Model(inputs=inputs, outputs=last)

tf.keras.utils.plot_model(model, show_shapes=True)

# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])


#show_predictions()

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset)