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

def not_random():
    seed = 0
    gs_model.random.seed(seed)


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


# test_vecs = []
# test = parse("The soldier hit you strongly.")
# for i in range(100):
#     gs_model.random.seed(0)
#     test_vecs.append(gs_model.infer_vector(test))
#
# best = np.mean(test_vecs, axis=0)
#
# for i in range(100):
#     print(spatial.distance.cosine(best, test_vecs[i]))
#     # print(gs_model.similarity(test_vecs[0], test_vecs[i]))
#     print(np.sum(np.square(best - test_vecs[i])))
# x_eval_text = parse("Your companion asks you for help.")
# print( x_eval_text)
# exit()
xx = []
yy = []

x_text = []
y_text = []

for task in pairs:
    question = parse(task[0])
    answer = parse(task[1])
    x_text.append(task[0])
    y_text.append(task[1])
    # print(question, answer)
    not_random()
    xx.append(gs_model.infer_vector(question))
    not_random()
    yy.append(gs_model.infer_vector(answer))
    # result = 1 - spatial.distance.cosine(x[-1], y[-1])
    # print(result)

# print(len(xx))


vec_size = len(xx[0])
print(np.mean(xx), np.mean(yy))
print(np.max(xx), np.min(yy))


# exit()

sess = tf.InteractiveSession()


x_layer = tf.placeholder(tf.float32, shape=[None, vec_size], name="x_layer")
y_layer = tf.placeholder(tf.float32, shape=[None, vec_size], name="y_layer")

Wh = tf.Variable(tf.random_normal([vec_size, vec_size]))
bh = tf.Variable(tf.random_normal([vec_size]))
hi = tf.matmul(x_layer, Wh) + bh
h = tf.nn.softsign(hi)

Wy = tf.Variable(tf.random_normal([vec_size, vec_size]))
by = tf.Variable(tf.random_normal([vec_size]))
yi = tf.matmul(h, Wy) + by
y = tf.nn.softsign(yi, name="y")


loss = tf.reduce_sum(tf.square(y - y_layer))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


# print(sess.run(bh))
loss_val = -1

for i in range(50000):
    _, loss_val = sess.run([train, loss], {x_layer: xx, y_layer: yy})
    print(loss_val)

print(loss_val)
y_placeholder = np.zeros([vec_size])
for i in range(10):
    _, loss_val, output = sess.run([train, loss, y], {x_layer: [xx[i]], y_layer: [y_placeholder]})
    print(i)
    print(loss_val)
    print(spatial.distance.cosine(output, yy[i]))
    print(np.sum(np.square(output - yy[i])))
    print(":::::")
    not_random()
    y_vec = gs_model.infer_vector(parse(y_text[i]))
    print(spatial.distance.cosine(yy[i], y_vec))
    print(np.sum(np.square(yy[i] - y_vec)))
    print("---------")

saver.save(sess, './rpgmodel')
# print(yy)
# "An orc asks you for help."

# x_eval_text = parse("A passer-by hit you.")
#
# x_eval = np.array([xx[1]], dtype=np.float32)
# print(np.mean(sess.run(y, {x_layer: x_eval, y_layer: yy})))
#
# x_eval = np.array([xx[16]], dtype=np.float32)
# print(np.mean(sess.run(y, {x_layer: x_eval, y_layer: yy})))
#
# print(np.mean(xx[1]), np.mean(yy[1]))
# print(np.mean(xx[16]), np.mean(yy[16]))
print(spatial.distance.cosine(yy[0], yy[1]))
print(np.sum(np.square(yy[0] - yy[1])))

print(spatial.distance.cosine(xx[0], yy[0]))
print(np.sum(np.square(xx[0] - yy[0])))

# print(na.answer("A snake bit you.", ["I do nothing.", "I drink a cure.", "I take rest.", "I figth him."]))
x_eval_text = parse("A snake bit you.")
not_random()
x_eval = np.array([gs_model.infer_vector(x_eval_text)], dtype=np.float32)
ideal = sess.run(y, {x_layer: x_eval, y_layer: [y_placeholder]})

for text in ["I do nothing.", "I drink a cure.", "I take rest.", "I figth him.", "I fight him."]:
    text = parse(text)
    print(text)
    not_random()
    ans = np.array([gs_model.infer_vector(text)], dtype=np.float32)
    print(np.mean(ans), np.mean(ideal))
    print(spatial.distance.cosine(ideal, ans))
    print(np.sum(np.square(ideal - ans)))

print("------------------------")

x_eval_text = parse("The soldier hit you strongly.")
not_random()
x_eval = np.array([gs_model.infer_vector(x_eval_text)], dtype=np.float32)
ideal = sess.run(y, {x_layer: x_eval, y_layer: [y_placeholder]})


for text in ["I do nothing.", "I drink a cure.", "I take rest.", "I figth him.", "I fight him."]:
    text = parse(text)
    print(text)
    not_random()
    ans = np.array([gs_model.infer_vector(text)], dtype=np.float32)
    print(ans.shape)
    print(ideal.shape)
    print(np.mean(ans), np.mean(ideal))
    print(spatial.distance.cosine(ideal, ans))
    print(np.sum(np.square(ideal - ans)))

# print(yy[16] - sess.run(y, {x_layer: x_eval, y_layer: yy}))

# print(sess.run(bh))



