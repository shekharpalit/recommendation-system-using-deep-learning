
from flask import Flask, render_template, url_for, request
import tensorflow as tf
import numpy as np
import pandas as pd

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    movies_df = pd.read_csv('ml-1m/movies.dat',sep='::', header = None)
    ratings_df = pd.read_csv('ml-1m/ratings.dat',sep='::', header = None)
    movies_df.columns = ['MovieID', 'Title', 'Genres']
    ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'TimeStamp']
    movies_df['List Index'] = movies_df.index
    merged_df = movies_df.merge(ratings_df, on='MovieID')#merge movie dataframe with ratings
    merged_df = merged_df.drop('TimeStamp', axis = 1).drop('Title', axis = 1).drop('Genres', axis = 1)
    user_Group = merged_df.groupby('UserID')

    #amount of users used for training

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
    # Current weight
    cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

    # Current visible unit biases
    cur_vb = np.zeros([visibleUnits], np.float32)

    # Current hidden unit biases
    cur_hb = np.zeros([hiddenUnits], np.float32)

    # Previous weight
    prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

    # Previous visible unit biases
    prv_vb = np.zeros([visibleUnits], np.float32)

    # Previous hidden unit biases
    prv_hb = np.zeros([hiddenUnits], np.float32)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    cur_w = np.load("W.npy")
    cur_vb = np.load("v_bias.npy")
    cur_hb = np.load("h_bias.npy")

    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        if comment == "Shekhar":
            inputUser = [trX[50]]
            hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
            vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
            feed = sess.run(hh0, feed_dict={v0: inputUser, W: cur_w, hb: cur_hb})
            rec = sess.run(vv1, feed_dict={hh0: feed, W: cur_w, vb: cur_vb})

                # List the 20 most recommended movies for our mock user by sorting it by their scores given by our model.
            scored_movies_df_50 = movies_df
            scored_movies_df_50["Recommendation Score"] = rec[0]
            res = scored_movies_df_50.sort_values(["Recommendation Score"], ascending=False).head(20)


    return render_template('view.html', recommendation = [res.to_html(classes='tab')], titles = ['na', 'Jhon', 'Xyz'])



if __name__ == '__main__':
    app.run(debug = True)
