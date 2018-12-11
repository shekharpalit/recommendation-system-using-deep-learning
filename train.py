import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%%matplotlib inline
movies_df = pd.read_csv('ml-1m/movies.dat',sep='::', header = None)
ratings_df = pd.read_csv('ml-1m/ratings.dat',sep='::', header = None)
movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'TimeStamp']
movies_df['List Index'] = movies_df.index
merged_df = movies_df.merge(ratings_df, on='MovieID')#merge movie dataframe with ratings
merged_df = merged_df.drop('TimeStamp', axis = 1).drop('Title', axis = 1).drop('Genres', axis = 1)
user_Group = merged_df.groupby('UserID')
#amount of users used for training


amountOfUsedUsers = 1000

# Creating the training list
trX = []

# For each user in the group
for userID, curUser in user_Group:

    # Create a temp that stores every movie's rating
    temp = [0]*len(movies_df)

    # For each movie in curUser's movie list
    for num, movie in curUser.iterrows():

        # Divide the rating by 5 and store it
        temp[movie['List Index']] = float(movie['Rating'])/5.0

    # Add the list of ratings into the training list
    trX.append(temp)

    # Check to see if we finished adding in the amount of users for training
    if amountOfUsedUsers == 0:
        break
    amountOfUsedUsers -= 1
trY = np.load("trX.npy")

trX = trY.tolist()
hiddenUnits = 50
visibleUnits = len(movies_df)
vb = tf.placeholder(tf.float32, [visibleUnits])  # Number of unique movies
hb = tf.placeholder(tf.float32, [hiddenUnits])  # Number of features were going to learn
W = tf.placeholder(tf.float32, [visibleUnits, hiddenUnits])  # Weight Matrix
# Phase 1: Input Processing
v0 = tf.placeholder("float", [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))  # Gibb's Sampling

# Phase 2: Reconstruction
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  # Hidden layer activation
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)
# Learning rate
alpha = 1.0

# Create the gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)

# Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

# Create methods to update the weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

# Set the error function, here we use Mean Absolute Error Function
err = v0 - v1
err_sum = tf.reduce_mean(err*err)
mae = tf.reduce_mean(tf.abs(err))
rmse = tf.sqrt(tf.reduce_mean(tf.square(err)))
# Train RBM with 15 Epochs, with Each Epoch using 10 batches with size 100 and print the RMSE and MSE
epochs = 15
batchsize = 100
errors = []
err1 = []
for i in range(epochs):
    for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
        err1.append(sess.run([mae, rmse], feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
        mean_average_error, root_mean_squared_error = (sess.run([mae, rmse], feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
        current_error = { "MAE": mean_average_error, "RMSE": root_mean_squared_error }
        errors.append(current_error)
    print ("MAE = {MAE:10.9f}, RMSE = {RMSE:10.9f}".format(**current_error))
plt.plot(err1)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()
#save the model in the local directory
np.save("W.npy", cur_w)
np.save("v_bias.npy", cur_vb)
np.save("h_bias.npy", cur_hb)
