from segmentation_models import Unet, Nestnet, Xnet, FPN
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import callbacks
import numpy as np
import warnings
from helper_functions import *
warnings.filterwarnings('ignore')

def simple(img_path='/media/dengpingfan/leone/dpfan/gepeng/Dataset/3Dataset/img',
           gt_path='/media/dengpingfan/leone/dpfan/gepeng/Dataset/3Dataset/gt',
           batchSize=7,
           target_size=(256, 256),
           epoch=24,
           lr = 0.03,
           steps_per_epoch= 663,
           model_save_path='./models/FPN/fpn_camouflage_baseline.h5'):
    seed = 1
    data_gen_args = dict(
        horizontal_flip=True,
        fill_mode='nearest')

    img_datagen = ImageDataGenerator(**data_gen_args)
    data_gen_args['rescale'] = 1./255
    mask_datagen = ImageDataGenerator(**data_gen_args)

    img_gen = img_datagen.flow_from_directory(
        img_path,
        batch_size=batchSize,
        target_size=target_size,
        shuffle=True,
        class_mode=None,
        seed=seed)

    mask_gen = mask_datagen.flow_from_directory(
        gt_path,
        color_mode='grayscale',
        batch_size=batchSize,
        target_size=target_size,
        shuffle=True,
        class_mode=None,
        seed=seed)
    train_gen = zip(img_gen, mask_gen)
    # model = Xnet(backbone_name='vgg16', encoder_weights='imagenet', decoder_block_type='transpose')
    model = FPN(backbone_name='resnet50', classes=1, activation='sigmoid')
    print(model.summary())
    opt = optimizers.SGD(lr=lr, momentum=0.9, decay=0.0001)
    model.compile(
        loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])

    save_best = callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor='loss',
        save_best_only=True,
        verbose=1)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=2, mode='min')
    callbacks_list = [save_best, early_stopping]

    model.fit_generator(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epoch,
        verbose=1,
        callbacks=callbacks_list)

if __name__ == '__main__':
    simple()