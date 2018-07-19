class Meter(object):
    def __init__(self):
        self.count = 0.
        self.total = 0.

    def update(self, v, count):
        self.count += count
        self.total += v

    def mean(self):
        return self.total / self.count

    def reset(self):
        self.count = 0
        self.total = 0
