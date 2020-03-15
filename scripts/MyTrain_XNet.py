from segmentation_models import Unet, Nestnet, Xnet
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import callbacks
import numpy as np
import warnings
from helper_functions import *
warnings.filterwarnings('ignore')


def simple(img_path=None,
           gt_path=None,
           batchSize=7,
           target_size=(256, 256),
           epoch=30,
           lr=3e-4,
           steps_per_epoch=None,
           model_save_path=None,
           seed=1):

    # ---- 1. prepare data ----
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
    # ---- 2. define your model ----
    model = Xnet(backbone_name='vgg16', encoder_weights='imagenet', decoder_block_type='transpose')
    print(model.summary())
    # ---- 3. define your optimizer ----
    opt = optimizers.Adam(lr=lr)
    model.compile(optimizer=opt,
                  loss=bce_dice_loss,
                  metrics=["binary_crossentropy", mean_iou, dice_coef])
    # ---- 4. snapshot ----
    save_best = callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor='loss',
        save_best_only=True,
        verbose=1)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='min')
    callbacks_list = [save_best, early_stopping]
    # ---- 5. start training ----
    model.fit_generator(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epoch,
        verbose=1,
        callbacks=callbacks_list)


if __name__ == '__main__':
    simple(img_path='/media/dengpingfan/leone/dpfan/gepeng/Dataset/3Dataset/img',
           gt_path='/media/dengpingfan/leone/dpfan/gepeng/Dataset/3Dataset/gt',
           batchSize=7,
           target_size=(256, 256),
           epoch=30,
           lr=3e-4,
           steps_per_epoch=663,
           model_save_path='./models/XNet/xnet_camouflage_baseline.h5')