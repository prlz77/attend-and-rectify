""" Harakiri helps models to kill themselves when they reach a plateau """
import sys


class Harakiri(object):
    """
    The main class
    """

    def __init__(self, initial_value=0):
        """ Constructor

        Args:
            initial_value: best value to compare with
        """
        self.best_value = initial_value
        self.buffer = []
        self.test = lambda: 1  # The test consists in two bits: plateau_length_exceeded current_value_is_best (2^1 2^0)
        self.message = "(Harakiri)"
        self.waypoints = []
        self.waypoint_callbacks = []
        self.plateau_callbacks = []

    def add_waypoint_callback(self, callback):
        """ Set a function to be called when a waypoint is violated

        Args:
            callback: a function reference
        """
        self.waypoint_callbacks.append(callback)

    def add_plateau_callback(self, callback):
        """ Set a function to be called when a plateau is violated

        Args:
            callback: a function reference
        """
        self.plateau_callbacks.append(callback)

    def add_waypoint(self, condition=lambda epoch, value: True):
        """ Add a condition that the training curve must accomplish. E.g. accuracy must always be higher than
            0 after the first epoch.

        Args:
            condition (function): tester function. Returns True to continue training.
        """
        self.waypoints.append(condition)

    def set_max_plateau(self, plateau_length):
        """ Sets harakiri to look for max values.

        Args:
            plateau_length: length of the plateau until seppuku
        """
        assert (plateau_length >= 1)
        self.test = lambda: 1 * (self.best_value > self.buffer[-1]) + 2 * (len(self.buffer) >= plateau_length)
        return self

    def set_min_plateau(self, plateau_length):
        """ Sets harakiri to look for max values

        Args:
            plateau_length: length of the plateau until seppuku
        """
        assert (plateau_length >= 1)
        self.test = lambda: 1 * (self.best_value < self.buffer[-1]) + 2 * (len(self.buffer) >= plateau_length)
        return self

    def set_message(self, message):
        """ Set the process last words

        Args:
            message: the last string of words
        """
        self.message = message

    def update(self, epoch, value):
        """ Tests the plateau condition given a new value

        Args:
            epoch (int or float): current epoch
            value (int or float): value to test
        """
        for waypoint in self.waypoints:
            if not waypoint(epoch, value):
                print(self.message, "Waypoint fail.")
                sys.exit(0)
        self.buffer.append(value)
        test_v = self.test()
        if test_v == 3:
            print(self.message, "Plateau reached.")
            sys.exit(0)
        elif test_v & 1 == 0:
            self.buffer = []
            self.best_value = value

