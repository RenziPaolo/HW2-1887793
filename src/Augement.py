


class Augment():
        def __init__(self, dataset, strategy):
            super(Augment, self).__init__()
            self.dataset = dataset
            self.strategy = strategy

        def apply(self):
            return self.dataset