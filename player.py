from manual_controller import ManualController
from nn_player import AutoController
from dialog import Dialog

class Player(object):
    def __init__(self):
        self.controller = AutoController()

    # wybierz odpowiedz na pytanie question z listy answers
    def answer(self, dialog):
        return self.controller.answer(dialog)

    def get_reward(self, reward):
        return self.controller.get_reward(reward)