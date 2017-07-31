"""
This file is used to use the predict the trained neural network and get
predictions for new input. The input an be given in the last line while
calling use_neural_network() method.

@Author Sanjay Haresh Khatwani (sxk6714@rit.edu)
@Author Savitha Jayasankar (skj9180@rit.edu)
@Author Saurabh Parekh (sbp4709@rit.edu)
"""

import create_feature_sets
import tensorflow as tf
import os


# Build the structure of the neural network exactly same as the
# train_and_test.py, so that the input features can be run through the neural
#  network.
number_nodes_HL1 = 100
number_nodes_HL2 = 100
number_nodes_HL3 = 100

x = tf.placeholder('float', [None, 23])
y = tf.placeholder('float')

with tf.name_scope("HiddenLayer1"):
    hidden_1_layer = {'number_of_neurons': number_nodes_HL1,
                      'layer_weights': tf.Variable(
                          tf.random_normal([23, number_nodes_HL1])),
                      'layer_biases': tf.Variable(
                          tf.random_normal([number_nodes_HL1]))}

with tf.name_scope("HiddenLayer2"):
    hidden_2_layer = {'number_of_neurons': number_nodes_HL2,
                      'layer_weights': tf.Variable(
                          tf.random_normal(
                              [number_nodes_HL1, number_nodes_HL2])),
                      'layer_biases': tf.Variable(
                          tf.random_normal([number_nodes_HL2]))}

with tf.name_scope("HiddenLayer3"):
    hidden_3_layer = {'number_of_neurons': number_nodes_HL3,
                      'layer_weights': tf.Variable(
                          tf.random_normal(
                              [number_nodes_HL2, number_nodes_HL3])),
                      'layer_biases': tf.Variable(
                          tf.random_normal([number_nodes_HL3]))}

with tf.name_scope("OutputLayer"):
    output_layer = {'number_of_neurons': None,
                    'layer_weights': tf.Variable(
                        tf.random_normal([number_nodes_HL3, 2])),
                    'layer_biases': tf.Variable(tf.random_normal([2])),}


# Nothing changes in this method as well.
def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['layer_weights']),
                hidden_1_layer['layer_biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['layer_weights']),
                hidden_2_layer['layer_biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['layer_weights']),
                hidden_3_layer['layer_biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['layer_weights']) + output_layer[
        'layer_biases']

    return output


saver = tf.train.Saver()


def use_neural_network(input_data):
    """
    In this method we restore the model created previously and obtain a
    prediction for an input sentence.
    :param input_data:
    :return:
    """
    prediction = neural_network_model(x)

    with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.join(os.getcwd(),
                                         'model\sarcasm_model.ckpt'))
        features = create_feature_sets.extractFeatureOfASentence(input_data)

        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x: [
            features]}), 1)))
        if result[0] == 0:
            print('Sarcastic:', input_data)
        elif result[0] == 1:
            print('Regular:', input_data)


# Supply the sentence to be tested below as a parameter in the method call.
if __name__ == '__main__':
    use_neural_network("Going to the gym surely makes you fit, in a same way standing in a garage make you a car!")
