class Array(object):
    """Default class for Arraysation function."""

    def __init__(self, data_array=None, hp=None):
        """Define the Arraysation function with the given hyperparameters hp."""

        self.data_array = data_array
        self.hp = hp
        self.name = data_array.__name__

    def call(self, X, Y):
        """Call the function Arraysation with the given hp."""

        return self.data_array(X, Y, **self.hp)
