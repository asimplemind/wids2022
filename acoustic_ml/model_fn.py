"""Models training and evaluating"""
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras import layers


MODELS = {
    'resnet50': ResNet50,
    'mobielnet': MobileNetV2
}


def compile_model(cfg: dict, model) -> Sequential:
    """ Compile model
    Args:
        cfg <dict>: dictionary containing configuration parameters
    Returns:
        model <keras.Sequential>: compiled keras model
    """

    # Depending on the number of classes, define loss function
    if cfg['model']['class_mode'] == 'binary':
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    else:
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['model']['learning_rate']),
        loss=loss,
        metrics=['acc'])

    # print(model.summary())

    return model


def build_model(cfg: dict) -> Sequential:
    """ Build model
    Args:
        cfg <dict>: dictionary containing configuration parameters
    Returns:
        model <keras.Sequential>: compiled keras model
    """

    if cfg['model']['type'] == 'built-in':
        print('Building and compiling built-in model\n')
        base_model = MODELS['resnet50'](input_shape=None,
                                        include_top=False,
                                        weights='imagenet',
                                        classes=None,  # cfg['data']['classes'],
                                        classifier_activation='softmax')

        # Make layers non-trainable
        fine_tune_at = cfg['model']['num_freeze_layers']
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        outputs = layers.Dense(cfg['model']['num_classes'], activation='sigmoid')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        print("Number of layers in the model: ", len(model.layers))

    else:
        # Another way to build a model using keras.Sequential()
        print('Using basic cnn model\n')

        # A basic 2 layers Conv2D, some dense layers
        model = Sequential()
        model.add(tf.keras.Input(shape=cfg['spectrogram']['image_shape']))
        model.add(layers.Conv2D(128, kernel_size=(5, 5), activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.SeparableConv2D(64, kernel_size=(5, 5), activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(cfg['model']['num_classes'], activation='softmax'))

    return model
