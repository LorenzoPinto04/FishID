import pandas as pd
from flask import Flask, jsonify, request, Response
import pickle
import base64
import jsonpickle
import numpy as np
import cv2
import json
from PIL import Image

# app
app = Flask(__name__)


prototxt = 'model/bvlc_googlenet.prototxt'
model = 'model/bvlc_googlenet.caffemodel'
labels = 'model/synset_words.txt'
# load the class labels from disk
rows = open(labels).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(prototxt, model)



# routes

@app.route('/', methods=['POST', 'GET'])
def predict():
    return 'Homepage Backend'


@app.route('/api/test', methods=['POST', 'GET'])
def test():
    try:
        if request.method == 'POST':

            r = request
            img = Image.open(r.files['file_field'])
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.imwrite('image.jpg', image)

            # our CNN requires fixed spatial dimensions for our input image(s)
            # so we need to ensure it is resized to 224x224 pixels while
            # performing mean subtraction (104, 117, 123) to normalize the input;
            # after executing this command our "blob" now has the shape:
            # (1, 3, 224, 224)
            blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
            
            # set the blob as input to the network and perform a forward-pass to
            # obtain our output classification
            net.setInput(blob)
            preds = net.forward()

            # sort the indexes of the probabilities in descending order (higher
            # probabilitiy first) and grab the top-5 predictions
            idxs = np.argsort(preds[0])[::-1][:50]
            listResults = []
            # loop over the top-5 predictions and display them
            for (i, idx) in enumerate(idxs):
                # draw the top prediction on the input image
                if i == 0:
                    text = "Label: {}, {:.2f}%".format(classes[idx],
                        preds[0][idx] * 100)
                    cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

                # display the predicted label + associated probability to the console
                output = ("{}, {}, {:.5}".format(i + 1,
                    classes[idx], preds[0][idx]))
                listResults.append(output)


            response = {'results' : listResults}
            response_pickled = jsonpickle.encode(response)
            return Response(response=response_pickled, status=200, mimetype="application/json")
        else:
            return ('[ERROR] La richiesta non è in POST')
    except Exception as e:
        response = {'Error' : str(e)}
        

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
