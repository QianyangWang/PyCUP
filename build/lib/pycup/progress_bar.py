import sys


class ProgressBar:

    def __init__(self,total_calcs):
        self.total = total_calcs

    def update(self,value):
        current_progress = self.calculate_current(value)
        print("\r", end="")
        print("Progress: {}%: ".format(round(current_progress,1)), "▋" * int(current_progress / 2), end="")
        sys.stdout.flush()

    def calculate_current(self,value):
        p = 100 * value/self.total
        return p