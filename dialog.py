class Dialog:
    def __init__(self, question, answers, end_happened):
        self.set(question, answers, end_happened)

    def set(self, question, answers, end_happened):
        self.question = question
        self.answers = answers
        self.end_happened = end_happened