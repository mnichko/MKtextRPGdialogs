from os import listdir
from os.path import isfile, join
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models.doc2vec import TaggedDocument
import gensim
# docLabels = [f for f in listdir("myDirPath") if f.endswith('.txt')]
#
# data = []
# for doc in docLabels:
#     data.append(open("myDirPath/" + doc, "r"))
#
#
# class LabeledLineSentence(object):
#     def __init__(self, doc_list, labels_list):
#         self.labels_list = labels_list
#         self.doc_list = doc_list
#
#     def __iter__(self):
#         for idx, doc in enumerate(self.doc_list):
#             print(doc)
#             yield TaggedDocument(doc.split(), tags=[self.labels_list[idx]])
# print("Start")
#
#
# # docLabels = [f for f in listdir("myDirPath") if f.endswith('.txt')]
# docLabels = ["input.txt"]
# data = []
# for doc in docLabels:
#     data.append(open(doc, 'r'))
#     print(data)
#
# it = LabeledLineSentence(data, [1])
# # print(it)= ["input.txt"]
# data = []
questions_path = "train_questionsq.txt"
answers_path = "train_answersq.txt"

questions = open(questions_path, 'r')
answers = open(answers_path, 'r')



doc = TaggedLineDocument("input.txt")

model = gensim.models.Doc2Vec(size=100, window=10, min_count=1, workers=11, alpha=0.025, min_alpha=0.025) # use fixed learning rate

model.build_vocab(doc)

model.iter = 300

model.train(doc, total_examples=model.corpus_count, epochs=model.iter)
#
# for epoch in range(10):
#     print(epoch)
#
#     model.alpha -= 0.002 # decrease the learning rate
#     model.min_alpha = model.alpha # fix the learning rate, no deca
#     # model.train(it)

model.save("modelgnsm")

# model = gensim.models.Doc2Vec.load("modelgnsm")
new_vector = model.infer_vector("boring , stupid or bad .")
new_vector = model.infer_vector("simplistic , silly and tedious .")
new_vector = model.infer_vector("good")
print(new_vector)
print(model.docvecs[1])
print(model.docvecs.doctags)
print(model.most_similar(positive=[new_vector]))
# print(model.most_similar("simplistic , silly and tedious ."))
# print(model["simplistic , silly and tedious ."])
print(doc)