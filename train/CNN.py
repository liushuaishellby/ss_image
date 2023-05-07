# -*- coding: UTF-8 -*-
import os
import warnings
import keras
import time
from keras import losses
from keras import optimizers
import tensorflow as tf
import keras.applications as cnn_model
import efficientnet.tfkeras as efn
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.optimizers import rmsprop_v2

# warnings.filterwarnings('ignore')
# # 忽略AVX2 FMA的警告
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == '__main__':
    # 以下是根据不同任务需求，需要进行手动修改的参数：
    # ————————————————————————————————————————————————————————————————————————————————————————————————
    # 选择要训练的卷积神经网络模型
    # 0 : VGG16
    # 1 : VGG19
    # 2 : ResNet50
    # 3 : ResNet101
    # 4 : ResNet152
    # 5 : MobileNet V1
    # 6 : MobileNet V2
    # 7 : DenseNet121
    # 8 : DenseNet169
    # 9 : DenseNet201
    # 10 : EfficientNet B0
    # 11 : EfficientNet B1
    # 12 : EfficientNet B2
    # 13 : EfficientNet B3
    # 14 : EfficientNet B4
    # 15 : EfficientNet B5
    # 16 : EfficientNet B6
    # 17 : EfficientNet B7
    # 18 : Xception
    # 19 : Inception V3
    # 20 : NASNetMobile
    # 21 : NASNetLarge
    CNN_name = ['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'ResNet152', 'MobileNet_V1', 'MobileNet_V2',
                'DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNet_B0', 'EfficientNet_B1',
                'EfficientNet_B2', 'EfficientNet_B3', 'EfficientNet_B4', 'EfficientNet_B5',
                'EfficientNet_B6', 'EfficientNet_B7', 'Xception', 'Inception_V3', 'NASNetMobile',
                'NASNetLarge']
    CNN_serial_number = 10

    # NB_CLASS: 分类的类别数量
    NB_CLASS = 100

    # 是否进行数据增强，0：不扩充数据，1：扩充数据
    augmentation_name = ['No_Aug', 'Aug']
    augmentation = 0

    # 根据使用的机器的显存大小，调整batch_size的值，一般为2的幂次。 如果出现报错，out of memory，那么可以把batch_size调小，比如16，8，4，一直到1为止。
    batch_size = 32

    # 总共的训练轮次
    EPOCH = 20

    # 多线程的数量，不支持的话换成1
    workers = 4

    # 以下是默认参数，如果对Keras不熟悉的话，采用默认值就可以：
    # ————————————————————————————————————————————————————————————————————————————————————————————————
    # 图片进行训练时，resize的大小
    IM_WIDTH = 224
    IM_HEIGHT = 224
    # 数据集路径
    train_root = 'final/train'
    validation_root = 'final/valid'
    train_root_augmentation = '.final/train_Augmentation'
    validation_root_augmentation = 'final/valid_Augmentation'
    # verbose：日志显示
    # verbose = 0 为不在标准输出流输出日志信息
    # verbose = 1 为输出进度条记录
    # verbose = 2 为每个epoch输出一行记录
    # 注意： 默认为 1
    verbose = 1

    # ——————————————————————————————————————————————————————————
    # 选择不同的损失函数,将需要使用的损失函数解除注释即可
    # 交叉熵损失函数，常用，默认使用这个即可
    loss = losses.categorical_crossentropy
    # loss = losses.mean_squared_error
    # loss = losses.mean_absolute_error
    # loss = losses.mean_absolute_percentage_error
    # loss = losses.mean_squared_logarithmic_error
    # loss = losses.squared_hinge
    # loss = losses.hinge
    # loss = losses.categorical_hinge
    # loss = losses.logcosh
    # loss = losses.sparse_categorical_crossentropy
    # loss = losses.binary_crossentropy
    # loss = losses.kullback_leibler_divergence
    # loss = losses.poisson
    # loss = losses.cosine_proximity
    # ——————————————————————————————————————————————————————————

    # ——————————————————————————————————————————————————————————
    # 选择不同的优化器,将需要使用的损失函数解除注释即可
    # sgd优化器，即随机梯度下降优化器。参数如下：
    # ——————————————————————————————
    # lr: float >= 0. 学习率。
    # momentum: float >= 0. 参数，用于加速 SGD 在相关方向上前进，并抑制震荡。
    # decay: float >= 0. 每次参数更新后学习率衰减值。
    # nesterov: boolean. 是否使用 Nesterov 动量。
    # ——————————————————————————————
    # optimizer = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    # RMSProp 优化器.参数如下：
    # ——————————————————————————————
    # lr: float >= 0. 学习率。
    # rho: float >= 0. RMSProp梯度平方的移动均值的衰减率.
    # epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
    # decay: float >= 0. 每次参数更新后学习率衰减值。
    # ——————————————————————————————
    # optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    # Adagrad 优化器.
    # Adagrad 是一种具有特定参数学习率的优化器，它根据参数在训练期间的更新频率进行自适应调整。
    # 参数接收的更新越多，更新越小。参数如下：    建议使用优化器的默认参数。
    # ——————————————————————————————
    # lr: float >= 0. 学习率.
    # epsilon: float >= 0. 若为 None, 默认为 K.epsilon().
    # decay: float >= 0. 每次参数更新后学习率衰减值.
    # ——————————————————————————————
    # optimizer = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

    # Adadelta优化器
    # Adadelta 是 Adagrad 的一个具有更强鲁棒性的的扩展版本，它不是累积所有过去的梯度，而是根据渐变更新的移动窗口调整学习速率。
    # 这样，即使进行了许多更新，Adadelta 仍在继续学习。 与 Adagrad 相比，在 Adadelta 的原始版本中，您无需设置初始学习率。
    # 在此版本中，与大多数其他 Keras 优化器一样，可以设置初始学习速率和衰减因子。 建议使用优化器的默认参数。
    # ——————————————————————————————
    # lr: float >= 0. 学习率，建议保留默认值。
    # rho: float >= 0. Adadelta梯度平方移动均值的衰减率。
    # epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
    # decay: float >= 0. 每次参数更新后学习率衰减值。
    # ——————————————————————————————
    # optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

    # Adam 优化器。常用的自适应优化器，参数如下：
    # ——————————————————————————————
    # lr: float >= 0. 学习率。
    # beta_1: float, 0 < beta < 1. 通常接近于 1。
    # beta_2: float, 0 < beta < 1. 通常接近于 1。
    # epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
    # decay: float >= 0. 每次参数更新后学习率衰减值。
    # amsgrad: boolean. 是否应用此算法的 AMSGrad 变种，来自论文 "On the Convergence of Adam and Beyond"。
    # ——————————————————————————————
    optimizer = adam_v2.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                        amsgrad=False)

    # Adamax优化器。来自 Adam 论文的第七小节.
    # 它是Adam算法基于无穷范数（infinity norm）的变种。 默认参数遵循论文中提供的值。参数如下：
    # ——————————————————————————————
    # lr: float >= 0. 学习率。
    # beta_1/beta_2: floats, 0 < beta < 1. 通常接近于 1。
    # epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
    # decay: float >= 0. 每次参数更新后学习率衰减值。
    # ——————————————————————————————
    # optimizer = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

    # Nesterov优化器，Nesterov 版本 Adam 优化器。
    # 正像 Adam 本质上是 RMSProp 与动量 momentum 的结合，
    # Nadam 是采用 Nesterov momentum 版本的 Adam 优化器。参数如下：
    # ——————————————————————————————
    # lr: float >= 0. 学习率。
    # beta_1/beta_2: floats, 0 < beta < 1. 通常接近于 1。
    # epsilon: float >= 0. 模糊因子. 若为 None, 默认为 K.epsilon()。
    # ——————————————————————————————
    # optimizer = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    # ————————————————————————————————————————————————————————————————————————————————————————————————
    # 数据生成器, 处理数据
    My_ImageDataGenerator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        rescale=1. / 255
    )
    # 如果augmentation为0，使用train_root和validation_root,
    # 如果为1，使用train_root_augmentation和validation_root_augmentation
    if augmentation is 0:
        if not os.path.exists(train_root):
            print('请检查:' + train_root + ' 路径中训练集是否存在')
            print('请检查:' + train_root + '  路径中验证集是否存在')
        train_generator = My_ImageDataGenerator.flow_from_directory(
            train_root,
            target_size=(IM_WIDTH, IM_HEIGHT),
            batch_size=batch_size,
            shuffle=True
        )
        vaild_generator = My_ImageDataGenerator.flow_from_directory(
            validation_root,
            target_size=(IM_WIDTH, IM_HEIGHT),
            batch_size=batch_size,
        )
    elif augmentation is 1:
        train_generator = My_ImageDataGenerator.flow_from_directory(
            train_root_augmentation,
            target_size=(IM_WIDTH, IM_HEIGHT),
            batch_size=batch_size,
            shuffle=True
        )
        vaild_generator = My_ImageDataGenerator.flow_from_directory(
            validation_root_augmentation,
            target_size=(IM_WIDTH, IM_HEIGHT),
            batch_size=batch_size,
        )
    else:
        print("请检查augmentation参数的值是否合理")

    # ————————————————————————————————————————————————————————————————————————
    # 按照不同的序号，构建不同的模型
    if CNN_serial_number is 0:
        base_model = cnn_model.vgg16.VGG16(weights='../pretrained/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                           include_top=False)
    elif CNN_serial_number is 1:
        base_model = cnn_model.vgg19.VGG19(weights='../pretrained/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                           include_top=False)
    elif CNN_serial_number is 2:
        base_model = cnn_model.resnet.ResNet50(
            weights='../pretrained/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number is 3:
        base_model = cnn_model.resnet.ResNet101(
            weights='../pretrained/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number is 4:
        base_model = cnn_model.resnet.ResNet152(
            weights='../pretrained/resnet152_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number is 5:
        base_model = cnn_model.mobilenet.MobileNet(weights='../pretrained/mobilenet_1_0_224_tf_no_top.h5',
                                                   include_top=False)
    elif CNN_serial_number is 6:
        base_model = cnn_model.mobilenet_v2.MobileNetV2(
            weights='../pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',
            include_top=False)
    elif CNN_serial_number is 7:
        base_model = cnn_model.densenet.DenseNet121(
            weights='../pretrained/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number is 8:
        base_model = cnn_model.densenet.DenseNet169(
            weights='../pretrained/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number is 9:
        base_model = cnn_model.densenet.DenseNet201(
            weights='../pretrained/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number is 10:
        base_model = efn.EfficientNetB0(
            weights='../pretrained/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
            include_top=False)
    elif CNN_serial_number is 11:
        base_model = efn.EfficientNetB1(
            weights='../pretrained/efficientnet-b1_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
            include_top=False)
    elif CNN_serial_number is 12:
        base_model = efn.EfficientNetB2(
            weights='../pretrained/efficientnet-b2_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
            include_top=False)
    elif CNN_serial_number is 13:
        base_model = efn.EfficientNetB3(
            weights='../pretrained/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
            include_top=False)
    elif CNN_serial_number is 14:
        base_model = efn.EfficientNetB4(
            weights='../pretrained/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
            include_top=False)
    elif CNN_serial_number is 15:
        base_model = efn.EfficientNetB5(
            weights='../pretrained/efficientnet-b5_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
            include_top=False)
    elif CNN_serial_number is 16:
        base_model = efn.EfficientNetB6(
            weights='../pretrained/efficientnet-b6_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
            include_top=False)
    elif CNN_serial_number is 17:
        base_model = efn.EfficientNetB7(
            weights='../pretrained/efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5',
            include_top=False)
    elif CNN_serial_number is 18:
        base_model = cnn_model.xception.Xception(
            weights='../pretrained/xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number is 19:
        base_model = cnn_model.inception_v3.InceptionV3(
            weights='../pretrained/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False)
    elif CNN_serial_number is 20:
        base_model = cnn_model.nasnet.NASNetMobile(weights='../pretrained/nasnet_mobile_no_top.h5',
                                                   include_top=False)
    elif CNN_serial_number is 21:
        base_model = cnn_model.nasnet.NASNetLarge(weights='../pretrained/nasnet_large_no_top.h5',
                                                  include_top=False)
    else:
        print("请检查CNN_serial_number参数的值是否合理")

    # 添加全局平均池化层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 添加一个全连接层
    x = Dense(256, activation='relu')(x)

    # 添加一个分类器，假设我们有NB_CLASS个类
    predictions = Dense(NB_CLASS, activation='softmax')(x)

    # 构建我们需要训练的完整模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # 迁移学习,只训练最后2个全连接层
    for layer in model.layers:
        layer.trainable = False
    for i in range(-2, 0):
        model.layers[i].trainable = True

    # 编译模型
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=['acc', keras.metrics.Precision(), keras.metrics.Recall()])
    print('模型编译完成，打印模型结构如下：')
    # 打印模型结构
    model.summary()

    # 日志存放路径
    # 获得当前时间时间戳
    now = int(time.time())
    # 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
    timeArray = time.localtime(now)
    otherStyleTime = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)
    log_dir = './logs/' + CNN_name[CNN_serial_number] + "_" + augmentation_name[augmentation] + "_" + otherStyleTime
    tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq="epoch", )

    # 最优模型的保存路径
    filePath = './' + CNN_name[CNN_serial_number] + "_" + augmentation_name[augmentation] + '_best.h5'
    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(filePath, monitor='val_acc', verbose=1, save_best_only=True, mode='max',
                                 save_freq="epoch")
    callbacks_list = [checkpoint, tbCallBack]

    # 在数据集上训练
    history = model.fit(train_generator, validation_data=vaild_generator,
                        epochs=EPOCH,
                        steps_per_epoch=train_generator.n / batch_size,
                        validation_steps=vaild_generator.n / batch_size, verbose=verbose,
                        callbacks=callbacks_list, workers=workers)
    print('模型训练完成！')
