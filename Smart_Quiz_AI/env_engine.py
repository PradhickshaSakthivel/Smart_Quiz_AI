from enum import Enum
import numpy as np

class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3

class EnvEngine:
    def __init__(self, initial="MEDIUM"):
        self.level = Difficulty[initial].value
        self.history = []

    def update(self, correct: bool):
        self.history.append(correct)

        if len(self.history) >= 3:
            recent = self.history[-3:]
            score = sum(recent) / 3

            if score > 0.7:
                self.level = min(3, self.level + 1)
            elif score < 0.3:
                self.level = max(1, self.level - 1)

    def get_level(self):
        return Difficulty(self.level).name
