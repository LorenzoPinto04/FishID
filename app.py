import pandas as pd
from flask import Flask, jsonify, request, Response
import pickle
import base64
import jsonpickle
import numpy as np
import cv2
import json

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
    # get data
    return strimg
    data = request.get_json(force=True)
    return data
    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return dat
    return jsonify(results=output)


    # route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    if request.method == 'POST':
    
        r = request
        nparr = np.fromstring(r.data, np.uint8)
        # decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #cv2.imwrite('file.jpg', image)
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
        idxs = np.argsort(preds[0])[::-1][:5]

        # loop over the top-5 predictions and display them
        for (i, idx) in enumerate(idxs):
            # draw the top prediction on the input image
            if i == 0:
                text = "Label: {}, {:.2f}%".format(classes[idx],
                    preds[0][idx] * 100)
                cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

            # display the predicted label + associated probability to the
            # console
            output = ("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
                classes[idx], preds[0][idx]))

        response = {'message': text}
        #response = {'message': 'image received'}
        # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)
        #return (response)
        return Response(response=response_pickled, status=200, mimetype="application/json")

        # convert string of image data to uint8
        # build a response dict to send back to client
        response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                    }
        #response = {'message': 'image received'}
        # encode response using jsonpickle
        response_pickled = jsonpickle.encode(response)
        #return (response)
        return Response(response=response_pickled, status=200, mimetype="application/json")
    else:
        return ('richiesta non in post')





if __name__ == '__main__':
    app.run(port = 5000, debug=True)
