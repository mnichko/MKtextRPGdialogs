from os import listdir
from os.path import isfile, join
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models.doc2vec import TaggedDocument
import gensim
from tf_learn_model import parse

model = gensim.models.Doc2Vec.load("modelgnsm")
docvecs = model.docvecs


print(docvecs.most_similar(12))

new_vector = model.infer_vector(['boring', 'tedious', 'bad', 'ugly', '.'])
print(len(new_vector))
print(docvecs.most_similar([new_vector]))
# new_vector = model.infer_vector("simplistic , silly and tedious .")
# new_vector = model.infer_vector("good")
# print(new_vector)
# print(model.docvecs[1])
# print(model.docvecs.doctags)
# print(model.most_similar(positive=[new_vector]))