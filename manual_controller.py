from dialog import Dialog


def is_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False


class ManualController(object):
    def __init__(self):
        pass

    def answer(self, dialog):
        print(dialog.question)
        print("----")
        if len(dialog.answers) == 0:
            return
        for i, answer in enumerate(dialog.answers):
            print(str(i+1)+":")
            print(answer)
        print("----")
        number = input("Proszę podać numer odpowiedzi:")
        try:
            number = int(number)
        except ValueError:
            pass
        while not is_int(number) or not 1 <= number <= len(dialog.answers):
            try:
                number = int(input("Błędny numer, proszę podać ponownie:"))
            except ValueError:
                number = -1

        return number

    def get_reward(self, reward):
        print("Nagroda:", reward)