import os
import sys
import json
import ConfigParser
"""
This is needed so that the script running on AWS will pick up the pre-compiled dependencies
from the vendored folder
"""
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(HERE, "vendored"))
"""
Now that the script knows where to look, we can safely import our objects
"""
# from tf_regression import TensorFlowRegressionModel
"""
Declare here global objects living across requests
"""
# use Pythonic ConfigParser to handle settings
Config = ConfigParser.ConfigParser()
Config.read(HERE + '/settings.ini')
# instantiate the tf_model in "prediction mode"
# tf_model = TensorFlowRegressionModel(Config, is_training=False)
# just print a message so we can verify in AWS the loading of dependencies was correct
print "loaded done!"

import tensorflow as tf

def validate_input(input_val):
    """
    Helper function to check if the input is indeed a float

    :param input_val: the value to check
    :return: the floating point number if the input is of the right type, None if it cannot be converted
    """
    try:
        float_input = float(input_val)
        return float_input
    except ValueError:
        return None


def get_param_from_url(event, param_name):
    """
    Helper function to retrieve query parameters from a Lambda call. Parameters are passed through the
    event object as a dictionary.

    :param event: the event as input in the Lambda function
    :param param_name: the name of the parameter in the query string
    :return: the parameter value
    """
    params = event['queryStringParameters']
    return params[param_name]


def return_lambda_gateway_response(code, body):
    """
    This function wraps around the endpoint responses in a uniform and Lambda-friendly way

    :param code: HTTP response code (200 for OK), must be an int
    :param body: the actual content of the response
    """
    return {"statusCode": code, "body": json.dumps(body)}













n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100
hm_epochs = 1


x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'f_fum':None,
                'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases':tf.Variable(tf.random_normal([n_classes])),}


def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

# saver = tf.train.Saver()
saver = tf.train.Saver();









def use_neural_network(input_data):
    prediction = neural_network_model(x)
    # with open('lexicon.pickle','rb') as f:
    #     lexicon = pickle.load(f)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess,"./model/model.ckpt")

        features = input_data

        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
        print("result", result)

        # otherResult = (sess.run(prediction.eval(feed_dict={x:[features]})))
        # print("OTHER result", other)
        return result[0];



















def predict(event, context):
    """
    This is the function called by AWS Lambda, passing the standard parameters "event" and "context"
    When deployed, you can try it out pointing your browser to

    {LambdaURL}/{stage}/predict?x=2.7

    where {LambdaURL} is Lambda URL as returned by serveless installation and {stage} is set in the
    serverless.yml file.

    """
    # otherValue = use_neural_network([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.45098039,  0.4745098 ,  0.63529412,  0.99215686,
    #     0.99215686,  0.83529412,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.24705882,  0.41960784,  0.66666667,  0.98431373,
    #     0.98823529,  0.98823529,  0.98823529,  0.98823529,  0.98039216,
    #     0.83921569,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.09803922,  0.75294118,  0.88627451,  0.88627451,  0.94509804,
    #     0.98823529,  0.99215686,  0.79215686,  0.98823529,  0.98823529,
    #     0.98823529,  0.98823529,  0.98823529,  0.88235294,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.26666667,  0.8745098 ,  0.98823529,
    #     0.98823529,  0.98823529,  0.98823529,  0.98823529,  0.15294118,
    #     0.0745098 ,  0.15294118,  0.25490196,  0.87843137,  0.98823529,
    #     0.98823529,  0.71764706,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.72941176,  0.98823529,  0.98823529,  0.98823529,  0.96078431,
    #     0.42352941,  0.20784314,  0.        ,  0.        ,  0.        ,
    #     0.58823529,  0.98823529,  0.98823529,  0.8627451 ,  0.07843137,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.2745098 ,  0.94901961,  0.98823529,
    #     0.98823529,  0.87058824,  0.23137255,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.69803922,  0.98823529,
    #     0.98823529,  0.55294118,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.7254902 ,  0.98823529,  0.98823529,  0.76078431,  0.2627451 ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.06666667,
    #     0.35294118,  0.94117647,  0.98823529,  0.76078431,  0.2627451 ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.3254902 ,  0.80392157,
    #     0.74509804,  0.09411765,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.4745098 ,  0.98823529,  0.98823529,
    #     0.81960784,  0.09411765,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.30196078,
    #     0.96862745,  0.98823529,  0.97254902,  0.41568627,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.99215686,  0.98823529,  0.98823529,
    #     0.4       ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.5254902 ,
    #     1.        ,  0.99215686,  0.99215686,  0.15294118,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.02352941,  0.71764706,  0.99215686,  0.98823529,
    #     0.41960784,  0.00784314,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.03921569,  0.4       ,
    #     0.98823529,  0.99215686,  0.63921569,  0.0627451 ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.05098039,  0.65882353,  0.98823529,  0.98823529,  0.43137255,
    #     0.00784314,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.16078431,  0.98823529,
    #     0.98823529,  0.85098039,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.15686275,  0.60784314,  0.98823529,  0.83921569,  0.12156863,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.64705882,  0.98823529,
    #     0.98823529,  0.41568627,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.16862745,  0.70196078,  0.98823529,  0.58823529,  0.15294118,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.5372549 ,  0.98823529,
    #     0.86666667,  0.15294118,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.2627451 ,  0.98823529,  0.30980392,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    #     0.        ,  0.        ,  0.        ,  0.        ])

    try:
        # param = get_param_from_url(event, 'x')

            # x_val = validate_input(param)
        x_val = event["x"]
        if x_val:
            # value = tf_model.predict(7)
            # value = 7;
            # value = x_val;

            # HERE BELOW
            nnReturn = use_neural_network(eval(x_val));
        else:
            raise "Input parameter has invalid type: float expected"
    except Exception as ex:
        error_response = {
            'error_message': "Unexpected error",
            'stack_trace': str(ex)
        }
        return return_lambda_gateway_response(503, error_response)

    return return_lambda_gateway_response(200, {'nnReturn': nnReturn, 'event': event["x"]})



