import cv2
import numpy as np
from keras.models import model_from_json
import json
import time


class SimpsonClassifier(object):

    def __init__(self, json_path, weights_path, pic_size=64):

        self.weights_path = weights_path
        self.input_shape = (pic_size, pic_size, 3)
        self.map_characters = {0: 'abraham_grampa_simpson',
                               1: 'apu_nahasapeemapetilon',
                               2: 'bart_simpson',
                               3: 'charles_montgomery_burns',
                               4: 'chief_wiggum',
                               5: 'comic_book_guy',
                               6: 'edna_krabappel',
                               7: 'homer_simpson',
                               8: 'kent_brockman',
                               9: 'krusty_the_clown',
                               10: 'lisa_simpson',
                               11: 'marge_simpson',
                               12: 'milhouse_van_houten',
                               13: 'moe_szyslak',
                               14: 'ned_flanders',
                               15: 'nelson_muntz',
                               16: 'principal_skinner',
                               17: 'sideshow_bob'}

        self.num_classes = len(self.map_characters)

        # Load model json
        with open(json_path, 'r') as f:
            model_json = f.read()
        self.model = model_from_json(model_json)

        # load weights into new model
        self.model.load_weights(weights_path)
        print("Loaded model from disk")

    def run(self, file):
        '''
        Run the model on the input image. Return the prediction and probability
        in a dictionary
        '''
        # read image file
        img_size = 64

        #nparr = np.fromstring(file.read().decode('UTF-8'), np.uint8)

        img = cv2.imdecode(np.fromstring(
            file.read(), np.uint8), cv2.IMREAD_COLOR)
        time.sleep(1)
        print(img.size)

        #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_size, img_size)).astype('float32') / 255.
        img = np.expand_dims(img, axis=0)

        print(img.shape)

        y_pred = self.model.predict_classes(img)[0]
        y_pred_name = self.map_characters[y_pred]
        y_prob = round(self.model.predict_proba(img)[0][y_pred], 2)
        return {'y_pred': y_pred_name, 'y_prob': str(y_prob)}
