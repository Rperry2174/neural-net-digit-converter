# neural-net-digit-converter
neural network model for converting handwritten digits into numerical counterparts
This code can be used in conjunction with "serverless" in order to create an AWS Lambda function to convert a (processed) image of a handwritten digit into its numerical counterpart

Sets up AWS endpoint (using AWS API Gateway)
Connects the AWS Lambda function to that endpoint to process data recieved through "params"
What the function does:

Takes as an input-param an array of 784 pixel values derived from a pre-processed image (see other repo)
Uses trained Neural Network model to predict what number has been drawn

Why I created it:
I wanted to create an application that uses a Neural Network to convert a handwritten digit into its numerical counterpart
The neural network I created is trained on 60,000 images that are in a 28x28 (784) pixel format and needs input data to be consistent with that

How to use it:
Install serverless (a tool for packaging aws functions)
Run "serverless create -t aws-python"
Run "serverless deploy"
Send request to endpoint in form: AWS_ENDPOINT/predict?x=ARRAY_OF_PIXEL_VALUES

Response is a single number representing the predicted digit
