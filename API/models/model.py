import tensorflow as tf

MODEL_PARAMS =  {
        'number_of_characters' :70,
        'alphabet': "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}",
        'max_length' :1014,
        'num_classes': 3
}
def get_model_params():
    return MODEL_PARAMS
class CharacterLevelCNN(tf.keras.Model):
    def __init__(self,args):
        super(CharacterLevelCNN,self).__init__()
        self.drop_input = tf.keras.layers.SpatialDropout1D(rate=args['drop_input'])
        self.conv1=  tf.keras.Sequential([tf.keras.layers.Conv1D(256,kernel_size=7,activation="relu"),tf.keras.layers.MaxPool1D(3)])
        self.conv2 = tf.keras.Sequential([tf.keras.layers.Conv1D(256,kernel_size=7,activation="relu"),tf.keras.layers.MaxPool1D(3)])
        self.conv3 = tf.keras.Sequential([tf.keras.layers.Conv1D(256,kernel_size=3,activation="relu")])
        self.conv4 =  tf.keras.Sequential([tf.keras.layers.Conv1D(256,kernel_size=3,activation="relu")])
        self.conv5 = tf.keras.Sequential([tf.keras.layers.Conv1D(256,kernel_size=3,activation="relu")])
        self.conv6 = tf.keras.Sequential([tf.keras.layers.Conv1D(256,kernel_size=3,activation="relu"),tf.keras.layers.MaxPool1D(3)])

        self.fc1 = tf.keras.Sequential([tf.keras.layers.Dense(1024,activation="relu"),tf.keras.layers.Dropout(0.5)])
        self.fc2 =  tf.keras.Sequential([tf.keras.layers.Dense(1024,activation="relu"),tf.keras.layers.Dropout(0.5)])
        self.fc3 = tf.keras.layers.Dense(get_model_params()['num_classes'])

    def __call__(self,x):
        x = self.drop_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = tf.reshape(x,(x.shape[0],-1))
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)
