from keras.preprocessing.image import ImageDataGenerator
from segmentation_models import Unet, Nestnet, Xnet, FPN, PSPNet
from keras import optimizers
import cv2
import os
from helper_functions import *

test_dict = {"CAMO": '/media/dengpingfan/leone/dpfan/gepeng/Dataset/CAMO_Test/img',
             "COD10K": '/media/dengpingfan/leone/dpfan/gepeng/Dataset/COD10K/COD10K/img',
             "CPD1K": '/media/dengpingfan/leone/dpfan/gepeng/Dataset/CPD_Split/CPD1K/img',
             "CHAMELEON": '/media/dengpingfan/leone/dpfan/gepeng/Dataset/CHAMELEON/img'}


def Simple(weight_path='./models/FPN/fpn_camouflage_baseline.h5',
           img_path=test_dict["COD10K"],
           target_size=(384, 384),
           batch_size=1,
           save_path="./results/PSPNet/COD10K"):

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

    model = PSPNet(backbone_name='vgg16', classes=1, activation='sigmoid')
    model.load_weights(weight_path)
    # opt = optimizers.Adam(lr=3e-4)
    # model.compile(optimizer=opt,
    #               loss=bce_dice_loss,
    #               metrics=["binary_crossentropy", mean_iou, dice_coef])
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
    Simple()
