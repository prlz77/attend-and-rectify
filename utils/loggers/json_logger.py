# -*- coding: utf-8 -*-
# Author: prlz77 <pau.rodriguez at gmail.com>
# Date: 05/06/2017
"""
Logs model state dict list into json file.
"""

import json
import datetime
import os

class JsonLogger():
    """
    The main class.
    """
    def __init__(self, path, rand_folder=False, duration=False):
        """ Constructor.

        Args:
            path: json output path
            unique_folder: create a unique folder with the current datetime
        """
        if rand_folder:
            path = os.path.join(path, str(datetime.datetime.now()).replace(' ', '_'))
            os.makedirs(path)
        self.path = os.path.join(path, 'log.json')
        self.output = open(self.path, 'w+')
        self.output.write('[]')
        self.first_time = True
        self.duration = datetime.datetime.now() if duration else False

    def update(self, state):
        """ Update log status (outputs to file).

        Args:
            state (dict): the current model state.
        """
        if not (self.duration is False):
            state['duration'] = str(datetime.datetime.now() - self.duration)

        self.output.seek(self.output.tell()-1, 0)
        if not self.first_time:
            self.output.write(',')
        else:
            self.first_time = False
        json.dump(state, self.output)
        self.output.write("]")
        self.output.flush()
