import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import shutil

print("TensorFlow version:", tf.__version__)

tutorial = 'cifar10'

print('Running {}'.format(tutorial))

#def load_data(tutorial):
#  if tutorial == 'mnist':

#def build_model(tutorial):

existing_model = 'new'

def run(tutorial, existing_model, epochs):

  model_path = 'model_{}'.format(tutorial)

  if tutorial == 'mnist':
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,epochs=5)

    model.evaluate(x_test, y_test, verbose=2)
  elif tutorial == 'fashion_mnist':
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5)

  elif tutorial == 'cifar10':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = np.squeeze(tf.image.rgb_to_grayscale(x_train))
    x_test = np.squeeze(tf.image.rgb_to_grayscale(x_test))
    
    model = tf.keras.Sequential([tf.keras.layers.Rescaling(scale=1./255.0),
                                tf.keras.layers.Flatten(input_shape=(np.shape(x_train[0]))),
                                tf.keras.layers.Dense(128, activation='relu'), 
                                tf.keras.layers.Dense(64, activation='relu'),
                                tf.keras.layers.Dense(10)])

    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    if existing_model == 'existing':
      try:
        model = tf.keras.models.load_model(model_path)
        print('Loading pretrained model from: {}'.format(model_path))
      except Exception as e:
        print('Did not find pretrained model in: {}'.format(model_path))
        pass
    elif existing_model == 'clear':
      shutil.rmtree(model_path)

        
    history = model.fit(x_train, y_train, epochs=epochs)#100)

    model.evaluate(x_test, y_test, verbose=2)

    model.save(model_path)

    plt.plot(history.history['accuracy'])
    print('Done')

  else:
    print('No tutorial selected!')

if __name__ == '__main__':
  run(tutorial='cifar10', existing_model = 'existing', epochs=50)