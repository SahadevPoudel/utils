from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import pandas as pd
import csv
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
DATASET_PATH  = '/home/poudelas/Documents/pycharm/ksavir/data/test/'
IMAGE_SIZE    = (224, 224)
NUM_CLASSES   = 8
BATCH_SIZE    = 8  # try reducing batch size or freeze more layers if your GPU runs out of memory
FREEZE_LAYERS = 2  # freeze the first this many layers for training
NUM_EPOCHS    = 100

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input
                                   )


test_batches = train_datagen.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  class_mode='categorical',
                                                  classes=['dyed-lifted-polyps', 'dyed-resection-margins',
                                                          'esophagitis', 'normal-cecum', 'normal-pylorus',
                                                          'normal-z-line', 'polyps', 'ulcerative-colitis'],
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False)
model1 = load_model('/home/poudelas/Documents/pycharm/ksavir/train_file/resnetv4.h5')

# pred=model1.predict_generator(test_batches,verbose=1)
# predicted_class_indices=np.argmax(pred,axis=1)
#
# print(pred)
# print(predicted_class_indices)
#
# labels = (test_batches.class_indices)
# print(labels)
#
# labels = dict((v,k) for k,v in labels.items())
# print(labels)
#
# predictions = [labels[k] for k in predicted_class_indices]
#print(predictions)




print(model1.summary())

results = model1.evaluate_generator(generator=test_batches)
print(results[0])
print(results[1])
exit()
Y_pred = model1.predict_generator(test_batches,verbose=1)
y_pred = np.argmax(Y_pred, axis=-1)
print(Y_pred)
print('Confusion Matrix')
cm = confusion_matrix(test_batches.classes, y_pred)
cm_df = pd.DataFrame(cm,
                      index = ['dyed-lifted-polyps', 'dyed-resection-margins',
                                                          'esophagitis', 'normal-cecum', 'normal-pylorus',
                                                          'normal-z-line', 'polyps', 'ulcerative-colitis'],
                      columns = ['dyed-lifted-polyps', 'dyed-resection-margins',
                                                          'esophagitis', 'normal-cecum', 'normal-pylorus',
                                                          'normal-z-line', 'polyps', 'ulcerative-colitis'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
#plt.title('Classification Result of 5 class \nAccuracy:{0:.3f}'.format(results[1]))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
# pred=model1.predict_generator(test_batches,verbose=1)
# predicted_class_indices=np.argmax(pred,axis=1)
#
# labels = (test_batches.class_indices)
# print(labels)
# labels = dict((v,k) for k,v in labels.items())
# print(labels)
# predictions = [labels[k] for k in predicted_class_indices]
# print(predictions)
#
# filenames=test_batches.filenames
# print(filenames)
# results=pd.DataFrame({"Filename":filenames,
#                       "Predictions":predictions})
# results.to_csv("ksavir.csv",index=False)



#important

# i=10
# prediction = []
# for i in range(i):
#     pred = model1.predict_generator(test_batches, verbose=1)
#     prediction.append(pred)
#
# preds = np.mean(prediction,axis=0)
