import boto3
import json
import email
import sms_spam_classifier_utilities as util

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    s3 = boto3.client('s3')
    messageRaw = s3.get_object(Bucket=bucket, Key=key)

    emailObj = email.message_from_bytes(messageRaw['Body'].read())
    femail = emailObj.get('From')
    body = emailObj.get_payload()[0].get_payload()
    
    print(femail)
    print(body)

    sgEndpoint = 'sms-spam-classifier-mxnet-2021-12-12-19-37-03-876'
    sruntime = boto3.client('runtime.sagemaker')

    #model input formatiing
    vocabulary_length = 9013
    input_mail = [body.strip()]
    one_hot_encode_data = util.one_hot_encode(input_mail, vocabulary_length)

    preProcessedInputMail = util.vectorize_sequences(one_hot_encode_data, vocabulary_length)
    jdata = json.dumps(preProcessedInputMail.tolist())

    #check for spam
    modelResponse = sruntime.invoke_endpoint(EndpointName=sgEndpoint, ContentType='application/json', Body=jdata)
    result = json.loads(modelResponse["Body"].read())

    if result['predicted_label'][0][0] == 0:
        label = 'Ok'
    else:
        label = 'Spam'
    
    prediction = round(result['predicted_probability'][0][0], 4)
    prediction = prediction*100

    print("Spam: ",label)
    print("Prediction: ", prediction)