
class Meter(object):
    """
    Abstract meter class from which all other meters inherit
    """
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass
