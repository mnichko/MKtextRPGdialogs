from dialog import Dialog
import random

import sys
from tf_answer_model import NeuralAnswerer


class AutoController(object):
    def __init__(self):
        self.answerer = NeuralAnswerer()

    def answer(self, dialog):

        # print(dialog.question)
        if len(dialog.answers) == 0:
            return
        # for i, answer in enumerate(dialog.answers):

        # while not is_int(number) or not 1 <= number <= len(dialog.answers):
        number = self.answerer.answer(dialog.question, dialog.answers) + 1
        # number = random.a
        return number

    def get_reward(self, reward):
        pass
        # elf.last_reward = None