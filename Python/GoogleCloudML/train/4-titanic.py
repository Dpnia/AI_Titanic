import pandas as pd
import numpy as np

import os.path
import tensorflow as tf

from tensorflow.python.lib.io import file_io

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', 'input', 'Input Directory.')
flags.DEFINE_string('output_dir', 'output', 'Output Directory.')

output_file = 'gs://titanic-ai-ml/output/tensorflow.csv'

def run_training():
    t1 = file_io.read_file_to_string('gs://titanic-ai-ml/input/train.csv')
    t1_data = t1.splitlines()

    t1_list_data = []
    t1_list_data2 = []
    for data in t1_data:
        t1_list_data.append(data.decode("utf-8"))

    for data in t1_list_data:
        t1_list_data2.append(data.split(","))

    t1_list_data2 = np.array(t1_list_data2[1:])

    train_df = pd.DataFrame(t1_list_data2, columns=['PassengerId', 'Survived', 'Pclass', 'Family Name', 'Last Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])

    train_df[train_df[::] == ''] = np.NaN

    train_df["PassengerId"] = train_df["PassengerId"].astype("int64")
    train_df["Survived"] = train_df["Survived"].astype("int64")
    train_df["Pclass"] = train_df["Pclass"].astype("int64")
    train_df["Age"] = train_df["Age"].astype("float64")
    train_df["SibSp"] = train_df["SibSp"].astype("int64")
    train_df["Parch"] = train_df["Parch"].astype("int64")
    train_df["Fare"] = train_df["Fare"].astype("float64")

    t2 = file_io.read_file_to_string('gs://titanic-ai-ml/input/test.csv')
    t2_data = t2.splitlines()

    t2_list_data = []
    t2_list_data2 = []
    for data in t2_data:
        t2_list_data.append(data.decode("utf-8"))

    for data in t2_list_data:
        t2_list_data2.append(data.split(","))

    t2_list_data2 = np.array(t2_list_data2[1:])

    test_df = pd.DataFrame(t2_list_data2, columns=['PassengerId', 'Pclass', 'Family Name', 'Last Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked'])

    test_df[test_df[::] == ''] = np.NaN

    test_df["PassengerId"] = test_df["PassengerId"].astype("int64")
    test_df["Pclass"] = test_df["Pclass"].astype("int64")
    test_df["Age"] = test_df["Age"].astype("float64")
    test_df["SibSp"] = test_df["SibSp"].astype("int64")
    test_df["Parch"] = test_df["Parch"].astype("int64")
    test_df["Fare"] = test_df["Fare"].astype("float64")

    train_data = pd.DataFrame(train_df, columns=["PassengerId", "Survived","Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])
    test_data = pd.DataFrame(test_df, columns=["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

    median_age_train = train_data["Age"].median()
    train_data["Age"] = train_data["Age"].fillna(median_age_train)

    mst_frq_embarked_train = train_data["Embarked"].value_counts().index[0]
    train_data["Embarked"] = train_data["Embarked"].fillna(mst_frq_embarked_train)

    Ports = list(enumerate(np.unique(train_data['Embarked'])))
    Ports_dict = {name: i for i, name in Ports}
    train_data["Embarked"] = train_data["Embarked"].map(lambda x: Ports_dict[x]).astype(int)

    train_data["Gender"] = 0
    train_data["Gender"][train_data["Sex"] == 'male'] = 1

    median_age_test = test_data["Age"].median()
    test_data["Age"] = test_data["Age"].fillna(median_age_test)

    mst_frq_embarked_test = test_data["Embarked"].value_counts().index[0]
    test_data["Embarked"] = test_data["Embarked"].fillna(mst_frq_embarked_test)

    test_data["Embarked"] = test_data["Embarked"].map(lambda x: Ports_dict[x]).astype(int)

    test_data["Gender"] = 0
    test_data["Gender"][test_data["Sex"] == 'male'] = 1

    test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].median())

    test_data["Survived"] = np.NaN

    X_train = train_data[["Pclass", "Gender", "Age", "Fare", "Embarked", "SibSp", "Parch"]]
    y_train = train_data["Survived"]

    X_test = test_data[["Pclass", "Gender", "Age", "Fare", "Embarked", "SibSp", "Parch"]]
    y_test = test_data["Survived"]

    X_train = X_train.astype("float")

    x_data = np.array(X_train, dtype=np.float32)
    y_data = np.reshape(np.array(train_data["Survived"], dtype=np.float32), (891, 1))

    x_data_test = np.array(X_test, dtype=np.float32)


    X = tf.placeholder(tf.float32, name="X-input")
    Y = tf.placeholder(tf.float32, name="Y-input")

    # Wide network: Use more neurons in each layer.
    W1 = tf.Variable(tf.random_uniform([7, 10], -1.0, 1.0), name="weight1")
    W2 = tf.Variable(tf.random_uniform([10, 10], -1.0, 1.0), name="weight2")
    W3 = tf.Variable(tf.random_uniform([10, 1], -1.0, 1.0), name="weight3")

    b1 = tf.Variable(tf.zeros([10]), name="bias1")
    b2 = tf.Variable(tf.zeros([10]), name="bias2")
    b3 = tf.Variable(tf.zeros([1]), name="bias3")

    # Hypotheses
    with tf.name_scope("layer2") as scope:
        L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

    with tf.name_scope("layer3") as scope:
        L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)

    with tf.name_scope("layer2") as scope:
        hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)

    # Cost function
    with tf.name_scope("cost") as scope:
        cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1. - hypothesis))

    # Minimize cost.
    a = tf.Variable(0.1)
    with tf.name_scope("train") as scope:
        optimizer = tf.train.GradientDescentOptimizer(a)
        train = optimizer.minimize(cost)

    # Initializa all variables.
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for step in range(50001):
            sess.run(train, feed_dict={X: x_data, Y: y_data})

            if step % 1000 == 0:
                print(
                    step,
                    sess.run(cost, feed_dict={X: x_data, Y: y_data}),
                    sess.run(W1),
                    sess.run(W2)
                )

        # Test model
        correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Check accuracy
        print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],
                       feed_dict={X: x_data, Y: y_data}))
        print("Accuracy:", accuracy.eval({X: x_data, Y: y_data}))

        # Predict
        predict = sess.run(tf.floor(hypothesis + 0.5), feed_dict={X: x_data_test})
        print(predict)

        saver = tf.train.Saver()
        checkpoint_file = os.path.join(FLAGS.output_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=0)

    lst_predict = list(predict)
    lst_predict2 = []
    for i in lst_predict:
        lst_predict2.append(i[0])
    lst_predict2
    test_data["Survived"] = pd.Series(lst_predict2, dtype="int64")
    test_data = pd.DataFrame(test_data, columns=["PassengerId", "Survived"])
    # test_data.to_csv("tensorflow.csv", index=False)
    test_data2 = np.array(test_data)
    # file_io.write_string_to_file(output_file, test_data)

    with file_io.FileIO(output_file, mode='w') as fout:
        fout.write("PassengerId,Survived"+'\n')
        for i,j in test_data2:
            k=np.append(i,j)
            fout.write(",".join([str(s) for s in k]) + '\n')

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()