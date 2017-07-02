from storyteller import StoryTeller
from player import Player

class Main(object):
    def __init__(self):
        self.storyTeller = StoryTeller("dragon_encounter.xml")
        self.player = Player()
        self.storyTeller.about_story()
        end_happened = False
        while not end_happened:
            dialog = self.storyTeller.ask_question()
            end_happened = dialog.end_happened
            number = self.player.answer(dialog)
            print("===========")
            print(dialog.question)
            if len(dialog.answers) > 0:
                print(dialog.answers[number-1])
            print("===========")
            if not end_happened:
                reward = self.storyTeller.get_answer(number)
                self.player.get_reward(reward)
        print(self.storyTeller.end())


if __name__ == "__main__":
    Main()
