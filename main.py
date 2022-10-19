from cProfile import label
from email.mime import base
from msilib.schema import _Validation
import os
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 数据集所在文件夹
base_dir = './data/cats_and_dogs'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
# 训练集
trainCats_dir = os.path.join(train_dir,'cats')
trainCats_dir = os.path.join(train_dir,'dogs')
# 验证集
validation_cats_dir = os.path.join(validation_dir,'cats')
validation_cats_dir = os.path.join(validation_dir,'dogs')

# 构建模型  构建卷积神经网络模型
model = tf.keras.models.Sequential([
    # 得到32个特征图，h*w都是3*3的
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape = (64,64,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # 得到特征图
    # 为全连接层准备
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation='relu'),   #512个隐藏特征

    # 二分类sigmoid就够了
    tf.keras.layers.Dense(1,activation='sigmoid')   #得到一个值在0~1的范围内

])

model.summary()

# 配置训练器
model.compile(loss='binary_crossentropy',
                optimizer=Adam(lr=1e-4),
                metrics=['acc'])


# 数据预处理
# 读进来的数据会被自动转换成tensor(float32)格式，分别准备训练和验证
# 图像数据归一化（0-1）区间
# 把数据全部压缩到了0-1
train_datagen = ImageDataGenerator(rescale=1./255)  #数据生成器
test_datagen = ImageDataGenerator(rescale=1./255)
# 生成数据
train_generator = train_datagen.flow_from_directory(
    train_dir,  #文件夹路径
    target_size=(64,64),    #指定resize的大小
    batch_size=20,
    # 如果one-hot就是categorical，二分类用binary就可以
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(64,64),
    batch_size=20,
    class_mode='binary'
)

# 训练网络模型
history = model.fit_generator(
    train_generator,
    steps_per_epoch=200 ,  #4000 images = batch_size *steps  (要进行迭代多少次。)就是数据量除以batch_size
    epochs = 20,
    validation_data=validation_generator,
    validation_steps= 100,   #2000 images = batch_size *steps
    verbose=2
)

# 效果展示


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs,acc,'bo',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
plt.title('Traing and validation accuracy')

plt.figure()

plt.plot(epochs,loss,'bo',label='Training Loss')
plt.figure(epochs,val_loss,'b',label='Validation Loss')
plt.title('Traing and validation loss')
plt.legend()

plt.show()






