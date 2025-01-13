model = keras.models.Sequential()
ResNet50

# 卷積層
model.add(keras.layers.Conv2D(filters=32, kernel_size=3,
                             padding='same', # 確保卷積後的圖像大小和原來輸入的圖像大小保持一樣
                             activation='relu', # 非線性轉換
                             input_shape=(width, height,1)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3,
                             padding='same', activation='relu'))

# 池化層
model.add(keras.layers.MaxPool2D(pool_size=2))

# 卷積層
model.add(keras.layers.Conv2D(filters=64, kernel_size=3,
                             padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=64,kernel_size=3,
                             padding='same', activation='relu'))

# 池化層
model.add(keras.layers.MaxPool2D(pool_size=2))

# 卷積層
model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                             padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                             padding='same', activation='relu'))

# 池化層
model.add(keras.layers.MaxPool2D(pool_size=2))

# 展平層
model.add(keras.layers.Flatten())

# 全連接層
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

# 編譯
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              )
model.summary()

epochs=50
train_num = train_generator.samples
valid_num = valid_generator.samples

valid_generator = None

def create_valid_generator():
    valid_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    )
    global valid_generator
    valid_generator = valid_datagen.flow_from_directory(
        val_folder,
        target_size=(height, width),
        batch_size=batch_size,
        seed=7,
        shuffle=False,
        class_mode='categorical'
    )

create_valid_generator()

history=model.fit_generator(
    train_generator,
    steps_per_epoch=train_num//batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_num//batch_size
    )