import tensorflow as tf
from tf_answer_learn import model
from scipy import spatial
import gensim
import numpy as np
from tf_answer_learn import parse

class NeuralAnswerer(object):
    def __init__(self):
        self.estimator = None
        self.gs_model = None

    def load_model(self):
        self.estimator = tf.contrib.learn.Estimator(model_fn=model, model_dir="model1.tf")
        self.gs_model = gensim.models.Doc2Vec.load("modelgnsm")


    def answer(self, question, answers):
        print(question)
        question = parse(question)
        question = self.gs_model.infer_vector(question)
        question = np.array([question], dtype=np.float32)
        print(question)
        # ideal = self.estimator.predict(x=question, batch_size=1)
        vec_size = len(question)

        # def input_fn():
        #     # feature_cols = {k: tf.constant(question[k].values)
        #     #                                for k in question}
        #     k = "x"
        #     feature_cols = {tf.contrib.layers.real_valued_column(k)}
        #     labels = tf.constant(['x'])
        #     return feature_cols, labels
        # y_eval = np.array([question], dtype=np.float32)
        y_eval = question
        input_fn = tf.contrib.learn.io.numpy_input_fn(
            {"x": question}, y_eval, batch_size=1, num_epochs=10)#, y_eval, batch_size=vec_size, num_epochs=10)
        # input_fn = tf.contrib.learn.io.numpy_input_fn(
        #     {"x": np.array([question])},y= np.array([question], dtype=np.float32), batch_size=1, num_epochs=10)
        ideal = self.estimator.predict(input_fn=input_fn)
        for i in ideal:
            print(i)

        ideal = i





        # def mnist_direct_data_input_fn(features_np_dict, targets_np):
        #     features_dict = {k: tf.constant(v) for k, v in features_np_dict.items()}
        #     targets = None if targets_np is None else tf.constant(targets_np)
        #     return features_dict, targets
        #
        #
        #
        #
        # tensor_prediction_generator = self.estimator.predict(
        #     input_fn=lambda: mnist_direct_data_input_fn(
        #         dict(
        #             images=np.array([image_orig]),
        #             fake_targets=np.array([label_target]),
        #         ), None),
        #     outputs=['image_gradient_vs_fake_target'],
        # )

        minimum = 100000
        best_answer = "AAA"
        for answer in answers:

            answer_parsed = parse(answer)
            vec = self.gs_model.infer_vector(answer_parsed)
            # vec = self.estimator.predict(x=answer, batch_size=1)

            # vec = self.estimator.predict(input_fn=infn)
            print(vec, ideal)
            result = 1 - spatial.distance.cosine(vec, ideal)
            print(result, answer)
            if result < minimum:
                minimum = result
                best_answer = answer
        return best_answer


na = NeuralAnswerer()
na.load_model()
print(na.answer("A passer-by hit you.", ["I do nothing.", "I fight him", "I insult him"]))