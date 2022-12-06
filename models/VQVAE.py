import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tree
# from vaes.DataGen import DatasetGenerator
from DataGen import DatasetGenerator
import time
import datetime
import os
import argparse

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

try:
  import sonnet.v2 as snt
  tf.enable_v2_behavior()
except ImportError:
  import sonnet as snt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='parent directory of training dataset')
    args = parser.parse_args()
    return args


class ResidualStack(snt.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super(ResidualStack, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._layers = []
        for i in range(num_residual_layers):
            conv3 = snt.Conv2D(
                output_channels=num_residual_hiddens,
                kernel_shape=(3, 3),
                stride=(1, 1),
                name="res3x3_%d" % i)
            conv1 = snt.Conv2D(
                output_channels=num_hiddens,
                kernel_shape=(1, 1),
                stride=(1, 1),
                name="res1x1_%d" % i)
            self._layers.append((conv3, conv1))

    def __call__(self, inputs):
        h = inputs
        for conv3, conv1 in self._layers:
            conv3_out = conv3(tf.nn.relu(h))
            conv1_out = conv1(tf.nn.relu(conv3_out))
            h += conv1_out
        return tf.nn.relu(h)  # Resnet V1 style


class Encoder(snt.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super(Encoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._enc_1 = snt.Conv2D(
            output_channels=self._num_hiddens // 2,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_1")
        self._enc_2 = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_2")
        self._enc_3 = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="enc_3")
        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)

    def __call__(self, x):
        h = tf.nn.relu(self._enc_1(x))
        h = tf.nn.relu(self._enc_2(h))
        h = tf.nn.relu(self._enc_3(h))
        return self._residual_stack(h)


class Decoder(snt.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens,
                 name=None):
        super(Decoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self._dec_1 = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="dec_1")
        self._residual_stack = ResidualStack(
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)
        self._dec_2 = snt.Conv2DTranspose(
            output_channels=self._num_hiddens // 2,
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_2")
        self._dec_3 = snt.Conv2DTranspose(
            output_channels=3,
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_3")

    def __call__(self, x):
        h = self._dec_1(x)
        h = self._residual_stack(h)
        h = tf.nn.relu(self._dec_2(h))
        x_recon = self._dec_3(h)
        return x_recon


class VQVAEModel(snt.Module):
    def __init__(self, encoder, decoder, vqvae, pre_vq_conv1,
                 data_variance, name=None):
        super(VQVAEModel, self).__init__(name=name)
        self._encoder = encoder
        self._decoder = decoder
        self._vqvae = vqvae
        self._pre_vq_conv1 = pre_vq_conv1
        self._data_variance = data_variance

    def __call__(self, inputs, is_training):
        z = self._pre_vq_conv1(self._encoder(inputs))
        vq_output = self._vqvae(z, is_training=is_training)
        x_recon = self._decoder(vq_output['quantize'])
        recon_error = tf.reduce_mean((x_recon - inputs) ** 2) / self._data_variance
        loss = recon_error + vq_output['loss']
        return {
            'z': z,
            'x_recon': x_recon,
            'loss': loss,
            'recon_error': recon_error,
            'vq_output': vq_output,
        }

def main():
    args = parse_arguments()
    print(f"{tf.__version__=}")
    print(f"{snt.__version__=}")
    # tf.compat.v1.reset_default_graph()
    # tf.compat.v1.disable_eager_execution()

    generator = DatasetGenerator([0,1,2], batch_size=10000, dim=(32,32,32), n_channels=1, feature="steering", shuffle=False)
    X, steering_Y, throttle_Y = generator.process_training_dir('/mnt/h/BeamNG_DAVE2_racetracks_all/PID/training_images_industrial-racetrackstartinggate0')
    train_data_dict = tree.map_structure(lambda x: x[:3000], X)
    #X_valid, steering_Y_valid, throttle_Y_valid = generator.process_training_dir('/mnt/h/BeamNG_DeepBillboard_dataset2/training_runs_hirochi_raceway-startingline_7755samples-S9SIPZ_YES')
    # X_valid, steering_Y_valid, throttle_Y_valid = generator.process_training_dir('/mnt/h/BeamNG_DeepBillboard_dataset2/training_runs_utah-westhighway2_25000samples-EEVXUR_YES')
    X_valid = generator.process_img_dir('/mnt/h/GitHub/superdeepbillboard/simulation/sampledir')
    print(f"{len(X_valid)=}")
    plt.imshow(X_valid[0])
    plt.pause(3)
    plt.savefig("sanity-check.jpg")
    # valid_data_dict = tree.map_structure(lambda x: x[3000:6000], X_valid)
    valid_data_dict = tree.map_structure(lambda x: x, X_valid)
    test_data_dict = tree.map_structure(lambda x: x[6000:], X)
    def cast_and_normalise_images(images):
        """Convert images to floating point with the range [-0.5, 0.5]"""
        X = (tf.cast(images, tf.float32) / 255.0) - 0.5
        return X

    train_data_variance = np.var(X / 255.0)
    print('train data variance: %s' % train_data_variance)
    print('validation data variance: %s' % np.var(X_valid / 255.0))
    batch_size = 32
    image_size = 32
    decay = 0.99
    learning_rate = 3e-4
    num_training_updates = 100
    embedding_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25

    image_size=(135,240)
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2

    # # Data Loading.
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_data_dict)
            .map(cast_and_normalise_images)
            .shuffle(1000)
            .repeat(-1)  # repeat indefinitely
            .batch(batch_size, drop_remainder=True)
            .prefetch(-1))

    valid_dataset = (
        tf.data.Dataset.from_tensor_slices(valid_data_dict)
            .map(cast_and_normalise_images)
            .repeat(1)  # 1 epoch
            .batch(batch_size)
            .prefetch(-1))

    # # Build modules.
    encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
    decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens)
    pre_vq_conv1 = snt.Conv2D(output_channels=embedding_dim,
                              kernel_shape=(1, 1),
                              stride=(1, 1),
                              name="to_vq")
    vq_use_ema = False
    if vq_use_ema:
        vq_vae = snt.nets.VectorQuantizerEMA(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost,
            decay=decay)
    else:
        vq_vae = snt.nets.VectorQuantizer(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost)

    model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1, data_variance=train_data_variance)

    optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

    checkpoint_root = "/mnt/h/GitHub/DAVE2-Keras/checkpoints"
    checkpoint_name = "example"
    save_prefix = os.path.join(checkpoint_root, checkpoint_name)

    # # A `Checkpoint` object manages checkpointing of the TensorFlow state associated
    # # with the objects passed to it's constructor. Note that Checkpoint supports
    # # restore on create, meaning that the variables of `my_module` do **not** need
    # # to be created before you restore from a checkpoint (their value will be
    # # restored when they are created).
    checkpoint = tf.train.Checkpoint(module=model)

    # # Most training scripts will want to restore from a checkpoint if one exists. This
    # # would be the case if you interrupted your training (e.g. to use your GPU for
    # # something else, or in a cloud environment if your instance is preempted).
    latest = tf.train.latest_checkpoint(checkpoint_root)
    if latest is not None:
        checkpoint.restore(latest)


    @tf.function
    def train_step(data):
        with tf.GradientTape() as tape:
            model_output = model(data, is_training=True)
        trainable_variables = model.trainable_variables
        grads = tape.gradient(model_output['loss'], trainable_variables)
        optimizer.apply(grads, trainable_variables)
        return model_output

    train_losses = []
    train_recon_errors = []
    train_perplexities = []
    train_vqvae_loss = []

    # for step_index, data in enumerate(train_dataset):
    #     data = tf.image.resize_with_pad(data, 152, 200)
    #     train_results = train_step(data)
    #     train_losses.append(train_results['loss'])
    #     train_recon_errors.append(train_results['recon_error'])
    #     train_perplexities.append(train_results['vq_output']['perplexity'])
    #     train_vqvae_loss.append(train_results['vq_output']['loss'])

    #     if (step_index + 1) % 100 == 0:
    #         print('%d train loss: %f ' % (step_index + 1,
    #                                       np.mean(train_losses[-100:])) +
    #               ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
    #               ('perplexity: %.3f ' % np.mean(train_perplexities[-100:])) +
    #               ('vqvae loss: %.3f' % np.mean(train_vqvae_loss[-100:])))
    #         checkpoint.save(save_prefix)
    #     if step_index == num_training_updates:
    #         break
    # checkpoint.save(save_prefix)

    # Reconstructions
    train_batch = next(iter(train_dataset))
    valid_batch = next(iter(valid_dataset))

    # Put data through the model with is_training=False, so that in the case of
    # using EMA the codebook is not updated.
    train_batch = tf.image.resize_with_pad(train_batch, 152, 200)
    train_reconstructions = model(train_batch, is_training=False)['x_recon'].numpy()
    valid_batch = tf.image.resize_with_pad(valid_batch, 152, 200)
    valid_reconstructions = model(valid_batch, is_training=False)['x_recon'].numpy()

    def convert_batch_to_image_grid(image_batch):
      reshaped = image_batch.reshape(4, 8, 152, 200, 3)
      try:
        reshaped = reshaped.transpose((0, 2, 1, 3, 4))
      except:
          reshaped = tf.transpose(reshaped, perm=[0, 2, 1, 3, 4])
      reshaped = reshaped.reshape(4 * 152, 8 * 200, 3)
      reshaped = (reshaped)
      return reshaped + 0.5

    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(2,2,1)
    ax.imshow(convert_batch_to_image_grid(train_batch.numpy()), interpolation='nearest')
    ax.set_title('training data originals')
    plt.axis('off')

    ax = f.add_subplot(2,2,2)
    ax.imshow(convert_batch_to_image_grid(train_reconstructions), interpolation='nearest')
    ax.set_title('training data reconstructions')
    plt.axis('off')
    
    ax = f.add_subplot(2,2,3)
    ax.imshow(convert_batch_to_image_grid(valid_batch.numpy()), interpolation='nearest')
    ax.set_title('validation data originals')
    plt.axis('off')
    
    ax = f.add_subplot(2,2,4)
    ax.imshow(convert_batch_to_image_grid(valid_reconstructions), interpolation='nearest')
    ax.set_title('validation data reconstructions')
    plt.axis('off')
    plt.savefig(f'VAE-{num_training_updates}epochs-{len(X)}samples-adversarialvalidation.jpg')


def reconstruct():
    model = instantiate_VQVAE()
    image_dims = (136, 240)
    batch_size=32
    generator = DatasetGenerator([0,1,2], batch_size=10000, dim=(32,32,32), n_channels=1, feature="steering", shuffle=False)
    X = generator.process_img_dir('C:/Users/Meriel/Documents/BeamNG_DeepBillboard_dataset2/training_runs_industrial-racetrackstartinggate_1685samples-4F71PP', size=image_dims)
     # = generator.process_img_dir('/mnt/h/BeamNG_DeepBillboard_dataset2/training_runs_hirochi_raceway-startingline_7755samples-S9SIPZ_YES')
    X_valid = generator.process_img_dir('C:/Users/Meriel/Documents/BeamNG_DeepBillboard_dataset2/training_runs_hirochi_raceway-startingline_7755samples-S9SIPZ_YES', size=image_dims)
    train_data_dict = tree.map_structure(lambda x: x, X)
    valid_data_dict = tree.map_structure(lambda x: x, X_valid)
    # test_data_dict = tree.map_structure(lambda x: x, X_test)
    def cast_and_normalise_images(images):
        """Convert images to floating point with the range [-0.5, 0.5]"""
        X = (tf.cast(images, tf.float32) / 255.0) - 0.5
        return X
    # # Data Loading.
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_data_dict)
            .map(cast_and_normalise_images)
            .shuffle(1000)
            .repeat(-1)  # repeat indefinitely
            .batch(batch_size, drop_remainder=True)
            .prefetch(-1))

    valid_dataset = (
        tf.data.Dataset.from_tensor_slices(valid_data_dict)
            .map(cast_and_normalise_images)
            .repeat(1)  # 1 epoch
            .batch(batch_size)
            .prefetch(-1))
    # Reconstructions
    train_batch = next(iter(train_dataset))
    valid_batch = next(iter(valid_dataset))

    # Put data through the model with is_training=False, so that in the case of
    # using EMA the codebook is not updated.
    train_reconstructions = model(train_batch, is_training=False)['x_recon'].numpy()
    valid_reconstructions = model(valid_batch, is_training=False)['x_recon'].numpy()


    def convert_batch_to_image_grid(image_batch):
      reshaped = (image_batch.reshape(4, 8, *image_dims, 3)
                  .transpose(0, 2, 1, 3, 4)
                  .reshape(4 * image_dims[0], 8 * image_dims[1], 3))
      return reshaped + 0.5


    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(2,2,1)
    ax.imshow(convert_batch_to_image_grid(train_batch.numpy()),
              interpolation='nearest')
    ax.set_title('training data originals')
    plt.axis('off')

    ax = f.add_subplot(2,2,2)
    ax.imshow(convert_batch_to_image_grid(train_reconstructions),
              interpolation='nearest')
    ax.set_title('training data reconstructions')
    plt.axis('off')

    ax = f.add_subplot(2,2,3)
    ax.imshow(convert_batch_to_image_grid(valid_batch.numpy()),
              interpolation='nearest')
    ax.set_title('validation data originals')
    plt.axis('off')

    ax = f.add_subplot(2,2,4)
    ax.imshow(convert_batch_to_image_grid(valid_reconstructions),
              interpolation='nearest')
    ax.set_title('validation data reconstructions')
    plt.axis('off')
    plt.savefig("C:/Users/Meriel/Documents/GitHub/deeplearning-input-rectification/VQVAE-reconstruct.jpg")

def cast_and_normalise_images(images):
    """Convert images to floating point with the range [-0.5, 0.5]"""
    X = (tf.cast(images, tf.float32) / 255.0) - 0.5
    return X

def instantiate_VQVAE():
    embedding_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25

    image_size=(135,240)
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2

    # # Build modules.
    encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
    decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens)
    pre_vq_conv1 = snt.Conv2D(output_channels=embedding_dim, kernel_shape=(1, 1),
                              stride=(1, 1), name="to_vq")

    vq_vae = snt.nets.VectorQuantizer(
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost)

    model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1, data_variance=0.036)
    checkpoint_root = "C:/Users/Meriel/Documents/GitHub/deeplearning-input-rectification/models/weights/checkpoints136x240-origdataset"
    checkpoint = tf.train.Checkpoint(module=model)
    latest = tf.train.latest_checkpoint(checkpoint_root)
    print(f"{latest=}")
    if latest is not None:
        checkpoint.restore(latest)
    return model

if __name__ == "__main__":
    # main()
    reconstruct()
