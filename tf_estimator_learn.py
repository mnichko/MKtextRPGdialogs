import tensorflow as tf
import numpy as np
import gensim
from scipy import spatial


questions_path = "train_questionsq.txt"
answers_path = "train_answersq.txt"
questions = open(questions_path, 'r')
answers = open(answers_path, 'r')

pairs = zip(questions, answers)

gs_model = gensim.models.Doc2Vec.load("modelgnsm")

def parse(text):
    t = text.lower()
    t = t.replace(".", " .")
    t = t.replace("?", " ?")
    t = t.replace("!", " !")
    t = t.replace(",", " ,")
    t = t.replace(";", " ;")
    t = t.replace(":", " :")
    t = t.replace("-", " - ")
    t = t.replace("(", "( ")
    t = t.replace(")", " )")
    t = t.replace("/", " / ")
    t = t.replace("&", " & ")
    t = t.split()
    return t

x = []
y = []

for task in pairs:
    question = parse(task[0])
    answer = parse(task[1])
    print(question, answer)
    x.append(gs_model.infer_vector(question))
    y.append(gs_model.infer_vector(answer))
    result = 1 - spatial.distance.cosine(x[-1], y[-1])
    print(result)

print(len(x))


# tensorfolw part, finally

# x = [0, 1, 2, 3, 4, 5]
# y = [1, 2, 3, 4, 5, 6]

# x = [x[0]]
# y = [y[0]]
# y = [xx*2 for xx in x]
# print(x[0][0], y[0][0])
# vec_size = len(x[0])
vec_size = len(x[0])


# print(vec_size)
# print(x)
# exit()
def model(features, labels, mode):
    # Hidden layer
    Wh = tf.get_variable("Wh", [vec_size, vec_size], dtype=tf.float32)
    bh = tf.get_variable("bh", [vec_size], dtype=tf.float32)
    hi = tf.matmul(features['x'], Wh) + bh
    h = tf.nn.relu(hi)


    Wh2 = tf.get_variable("Wh2", [vec_size, vec_size], dtype=tf.float32)
    bh2 = tf.get_variable("bh2", [vec_size], dtype=tf.float32)
    hi2 = Wh*h + bh2
    h2 = tf.nn.relu(hi2)

    # Output layer
    Wy = tf.get_variable("Wy", [vec_size, vec_size], dtype=tf.float32)
    by = tf.get_variable("by", [vec_size], dtype=tf.float32)
    yi = Wy*h2+ by
    y = tf.nn.relu(yi)
    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))
    # Training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss),
               tf.assign_add(global_step, 1))
    # ModelFnOps connects subgraphs we built to the
    # appropriate functionality.
    return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)

# estimator = tf.contrib.learn.Estimator(model_fn=model, )
estimator = tf.contrib.learn.LinearEstimator(feature_columns=x)
# define our data sets
x_train = np.array(x, dtype=np.float32)
y_train = np.array(y, dtype=np.float32)
# x_eval_text = parse("Your companion asks you for help.")
x_eval_text = parse("An orc asks you for help.")
# y_eval_text = parse("I help him.")
# y_eval_text = parse("I begin to fight.")
y_eval_text = parse("I refuse to help.")
x_eval = np.array([gs_model.infer_vector(x_eval_text)], dtype=np.float32)
y_eval = np.array([gs_model.infer_vector(y_eval_text)], dtype=np.float32)

input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, vec_size, num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=vec_size, num_epochs=1000)

# train
for i in range(1, 2):
    estimator.fit(input_fn=input_fn, steps=100)
    # Here we evaluate how well our model did.
    train_loss = estimator.evaluate(input_fn=input_fn)
    eval_loss = estimator.evaluate(input_fn=eval_input_fn)
    print(i)
    print("train loss: %r" % train_loss)
    print("eval loss: %r" % eval_loss)


