import argparse
import tensorflow as tf
from PIL import Image
import numpy as np
from util import process_image
import tensorflow_hub as hub
import json

def predict(image_path, model, k=5):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    processed_image = processed_image[np.newaxis]
    
    predictions_array = model.predict(processed_image)
    predictions_array = np.squeeze(predictions_array)
    top_k = np.argpartition(-predictions_array, k)[:k]

    classes = []
    for i in top_k:
        classes.append(i)

    return predictions_array[top_k], classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('path',help='path to the image you want the model to predict')
    parser.add_argument('saved_model',help='The model you want to make a prediction on')
    parser.add_argument('--top_k',help='The top k predictions to show',type=int)
    parser.add_argument('--category_names',help='The path to JSON file to map the predictions')
    
    args = parser.parse_args()
    image_path = '{}'.format(args.path)
    model = tf.keras.models.load_model(args.saved_model,custom_objects={'KerasLayer':hub.KerasLayer})
    if args.top_k == None:
        probs, classes = predict(image_path,model,5)
    else:
        probs, classes = predict(image_path,model,k=args.top_k)
    
    if args.category_names != None:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        labeled_classes = []
        for i in classes:
            i = i + 1
            labeled_classes.append(class_names[str(i)])
        print(labeled_classes)
    print(probs)

        