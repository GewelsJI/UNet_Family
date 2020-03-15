from keras.preprocessing.image import ImageDataGenerator
from segmentation_models import Unet, Nestnet, Xnet
from keras import optimizers
import cv2
import os
from helper_functions import *


def Simple(weight_path=None,
           img_path=None,
           target_size=(256, 256),
           batch_size=1,
           save_path=None):
    os.makedirs(save_path, exist_ok=True)
    print(img_path)
    data_gen_args = dict(
        fill_mode='nearest')
    img_datagen = ImageDataGenerator(**data_gen_args)

    test_gen = img_datagen.flow_from_directory(
        img_path,
        batch_size=batch_size,
        target_size=target_size,
        shuffle=False,
        class_mode=None)

    model = Xnet(backbone_name='vgg16', encoder_weights='imagenet', decoder_block_type='transpose')
    print(model.summary())
    model.load_weights(weight_path)
    opt = optimizers.Adam(lr=3e-4)
    model.compile(optimizer=opt,
                  loss=bce_dice_loss,
                  metrics=["binary_crossentropy", mean_iou, dice_coef])
    predicted_list = model.predict_generator(test_gen,
                                             steps=None,
                                             max_queue_size=10,
                                             workers=1,
                                             use_multiprocessing=False,
                                             verbose=1)
    img_name_list = test_gen.order_filenames

    num = 0
    for predict in predicted_list:
        img_name = img_name_list[num].split('/')[1]
        img_name = img_name.replace('.jpg', '.png')
        img = Convert(predict)
        cv2.imwrite(os.path.join(save_path, img_name), img)
        num += 1
        print("[INFO] {}/{}".format(num, len(img_name_list)))


def Convert(predict):
    return (predict - predict.min()) / (predict.max() - predict.min()) * 255


if __name__ == '__main__':
    test_dict = {'dict_name': 'your_path'}

    Simple(weight_path='snapshots/save/pth',
           img_path=test_dict["dict_name"],
           target_size=(256, 256),
           batch_size=1,
           save_path="your/save/path")
