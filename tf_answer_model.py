import tensorflow as tf
from scipy import spatial
import gensim
import numpy as np
# from tf_learn_model import parse
# from tf_learn_model import yy

vec_size = 100
y_placeholder = np.zeros([vec_size])



def compare(vec_quest, vec_ans):
    return 1 - spatial.distance.cosine(vec_quest, vec_ans)

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


class NeuralAnswerer(object):
    def __init__(self):
        self.gs_model = None
        self.sess = None
        self.x_layer = None
        self.y_layer = None
        self.y = None
        self.load_model()

    def question2ideal_ans_vec(self, question):
        x_eval_text = parse(question)
        self.no_random()
        x_eval = np.array([self.gs_model.infer_vector(x_eval_text)], dtype=np.float32)
        return self.sess.run(self.y, {self.x_layer: x_eval, self.y_layer: [y_placeholder]})
        # return self.sess.run(self.y)

    def load_model(self):
        self.gs_model = gensim.models.Doc2Vec.load("modelgnsm")
        self.sess = tf.Session()
        # First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('./rpgmodel.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()
        self.x_layer = graph.get_tensor_by_name("x_layer:0")
        self.y_layer = graph.get_tensor_by_name("y_layer:0")
        self.y = graph.get_tensor_by_name("y:0")

    def no_random(self):
        # self.gs_model
        seed = 0
        self.gs_model.random.seed(seed)

    def answer(self, question, answers, cosine=True, verbose=True):
        ideal = self.question2ideal_ans_vec(question)
        minimum = 100000
        best_answer = "AAA"
        best_id = -1
        # print(question)
        for i, answer in enumerate(answers):
            answer_parsed = parse(answer)
            self.no_random()
            ans_vec = self.gs_model.infer_vector(answer_parsed)
            if cosine:
                result = spatial.distance.cosine(ans_vec, ideal)
            else:
                result = np.sum(np.square(ans_vec - ideal))
            if verbose:
                print(answer_parsed, result)
            if result < minimum:
                minimum = result
                best_answer = answer
                best_id = i
        return best_id

    def verbose_answer(self, question, answers):
        # question = parse(question)
        # question = self.gs_model.infer_vector(question)
        # question = np.array([question], dtype=np.float32)
        ideal = self.question2ideal_ans_vec(question)
        print(question)

        minimum = 100000
        best_answer = "AAA"
        for answer in answers:

            answer_parsed = parse(answer)
            self.no_random()
            ans_vec = self.gs_model.infer_vector(answer_parsed)
            # vec = self.estimator.predict(x=answer, batch_size=1)
            # print(answer_parsed, ans_vec)
            # vec = self.estimator.predict(input_fn=infn)
            # print(vec, ideal)
            result = spatial.distance.cosine(ans_vec, ideal)
            result2 = np.sum(np.square(ans_vec - ideal))
            print(result, result2, answer)
            if result2 < minimum:
                minimum = result2
                best_answer = answer
        return best_answer


na = NeuralAnswerer()
na.load_model()
# print(na.verbose_answer("You see a blue dragon on the field to the right.", ["I do nothing.", "I fight him.", "I insult him.", "I run away.", "I go to the left."]))
# print(na.verbose_answer("A passer-by hit you.", ["I do nothing.", "I fight him.", "I insult him.", "I run away."]))
# print(na.verbose_answer("A blue dragon is on your way.", ["I do nothing.", "I fight him.", "I insult him.", "I run away.", "I go away."]))
# print(na.verbose_answer("You saw a red dragon.", ["I do nothing.", "I fight him.", "I insult him.", "I run away.", "I go away."]))
# print(na.verbose_answer("You saw a green dragon.", ["I do nothing.", "I fight him.", "I insult him.", "I run away.", "I go away."]))
# print(na.verbose_answer("You saw a wolf.", ["I do nothing.", "I fight him.", "I insult him.", "I run away.", "I go away."]))
# print(na.verbose_answer("You are tired.", ["I do nothing.", "I keep going.", "I take rest."]))
# print(na.verbose_answer("A snake bit you.", ["I do nothing.", "I drink a cure.", "I take rest.", "I fight him."]))
# print(na.verbose_answer("A snake bit you.", ["I do nothing.", "I drink a cure.", "I take rest.", "I fight him."]))
# print(na.verbose_answer("A snake bit you.", ["I do nothing.", "I drink a cure.", "I take rest.", "I fight him."]))
#
# print(na.answer("A bandit fights with a peasant.", ["I do nothing.", "I drink a cure.", "I take rest.", "I fight him.", "I try to help."]))
