import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))
# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    
    vgg_input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name);
    vgg_keep_prob_tensor =  graph.get_tensor_by_name(vgg_keep_prob_tensor_name);
    vgg_layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name);
    vgg_layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name);
    vgg_layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name);
    
    return vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor
    
tests.test_load_vgg(load_vgg, tf)

def conv2d_transpose(input_layer, num_classes, kernel = 2, stride = 2):
    """
    :param input_layer: The layer to input,
    :param num_classes:  The number of the classes
    :return: The transposed layer
    """
    return tf.layers.conv2d_transpose(input_layer, num_classes, kernel, strides=(stride, stride), padding='same', 
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # 1x1
    conv_1x1 = conv2d_transpose(vgg_layer7_out, num_classes, kernel = 1, stride = 1)
    # 4x conv7 layer (32/4 = 8):
    output = conv2d_transpose(conv_1x1, num_classes, kernel = 8, stride = 4)
    print('original conv7 shape: ', vgg_layer7_out.shape)
    print('4x conv7 shape: ', output.shape)
    # 2x conv4 layer:
    conv_4 = conv2d_transpose(vgg_layer4_out, num_classes, kernel = 4, stride = 2)
    print('2x conv4 shape: ', conv_4.shape)
    output = tf.add(output, conv_4)

    # 1x conv3 layer:
    conv_3 = conv2d_transpose(vgg_layer3_out, num_classes, kernel = 1, stride = 1)
    print('original conv3 shape: ', vgg_layer3_out.shape)
    print('1x conv3 shape: ', conv_3.shape)
    output = tf.add(output, conv_3)

    #upscale the output to 8x now to match the input size
    output = conv2d_transpose(output, num_classes, kernel = 16, stride = 8)
    print('final output shape: ', output.shape)

    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(cross_entropy_loss)

    return logits, training_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image, correct_label: label,keep_prob: 0.5, learning_rate: 0.001})
            print("Loss: = {:.3f}".format(loss))
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    epochs = 5
    batch_size = 1
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        vgg_input_tensor, vgg_keep_prob_tensor, vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor = load_vgg(sess, vgg_path)
        layer = layers(vgg_layer3_out_tensor, vgg_layer4_out_tensor, vgg_layer7_out_tensor, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layer, correct_label, learning_rate, num_classes)
        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, vgg_input_tensor,
              correct_label, vgg_keep_prob_tensor, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob_tensor, vgg_input_tensor)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
