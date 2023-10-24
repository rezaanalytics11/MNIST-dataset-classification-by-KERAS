import keras
import tensorflow as ts
import numpy as np
import matplotlib.pyplot as plt

data=keras.datasets.mnist

(train_images,train_labels),(test_images,test_labels)=data.load_data()

train_images=train_images/255
test_images=test_images/255

print(test_labels.shape)

for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(train_images[i])
    plt.axis('off')
plt.show()

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history=model.fit(train_images, train_labels, epochs=5)



test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)

plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['loss'],label='loss')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend(['accuracy','loss'])
plt.show()

predictions = model.predict(test_images)
print(predictions)
plt.figure(figsize=(5,5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel('[test_labels[i]]')
    plt.title('class_names[np.argmax(predictions[i])]')
    plt.show()

    print(np.argmax(predictions[i]))