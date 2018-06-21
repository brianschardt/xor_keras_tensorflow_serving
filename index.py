import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)
K.set_learning_phase(0)  # all new operations will be in test mode from now on

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

model_version = "2" #Change this to export different model versions, i.e. 2, ..., 7
epoch = 100 ## the higher this number is the more accurate the prediction will be 10000 is a good number to set it at just takes a while to train

#Exhaustion of Different Possibilities
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

#Return values of the different inputs
Y = np.array([[0],[1],[1],[0]])

#Create Model
model = Sequential()
model.add(Dense(8, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.1)

model.compile(loss='binary_crossentropy', optimizer=sgd)
model.fit(X, Y, batch_size=1, nb_epoch=epoch)

test = np.array([[0.0,0.0]])

#setting values for the sake of saving the model in the proper format
x = model.input
y = model.output

print 'Results of Model', model.predict_proba(X)

prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"inputs": x}, {"prediction":y})

valid_prediction_signature = tf.saved_model.signature_def_utils.is_valid_signature(prediction_signature)
if(valid_prediction_signature == False):
    raise ValueError("Error: Prediction signature not valid!")

builder = saved_model_builder.SavedModelBuilder('./'+model_version)
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

# Add the meta_graph and the variables to the builder
builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_signature,
      },
      legacy_init_op=legacy_init_op)

# save the graph
builder.save()