import xml.etree.cElementTree as ET
from dialog import Dialog


class StoryTeller(object):
    def __init__(self, file_name):
        self.tree = None
        self.root = None
        self.current_dialog_name = ""  # nazwa aktualnego dialogu
        self.reward_sum = 0
        self.end_happened = False
        self.open(file_name)


    def open(self, file_name):
        self.tree = ET.parse(file_name)
        self.root = self.tree.getroot()
        self.current_dialog_name = "start"  # zawsze zaczynamy od start  # może to zmienić na type="start"?
        self.reward_sum = 0
        self.end_happened = False

    # TODO: zabezpieczyć się przed złymi wartościami
    # answer == numer odpowiedzi
    def get_answer(self, id):
        new_reward = 0
        for xml_dialog in self.root.iter("dialog"):
            if xml_dialog.get("name") == self.current_dialog_name:
                for xml_answer in xml_dialog.iter("answer"):
                    if id == int(xml_answer.get("id")):
                        if xml_answer.get("reward") is not None:
                            new_reward = int(xml_answer.get("reward"))
                            self.reward_sum += new_reward
                        self.current_dialog_name = xml_answer.get("link")
                        return new_reward
        return new_reward

    def ask_question(self):
        for xml_dialog in self.root.iter("dialog"):
            # print(xml_dialog.get("name"), self.current_dialog_name)
            if xml_dialog.get("name") == self.current_dialog_name:
                question = xml_dialog.find("question").text
                question = question.strip()
                answers = []

                if xml_dialog.get("type") == "end":
                    self.end_happened = True
                # count = 0
                # Nie wiem, czy tak ma zostać, ale będzie działać gdy jest nie po kolei
                for xml_answer in xml_dialog.iter("answer"):
                    answers.append("")
                for xml_answer in xml_dialog.iter("answer"):
                    id = int(xml_answer.get("id"))
                    text = xml_answer.text
                    answers[id-1] = text
                dialog = Dialog(question, answers, self.end_happened)
                return dialog

    def end(self):
        return self.reward_sum

    def about_story(self):
        print(self.root.find("about").text.strip())

