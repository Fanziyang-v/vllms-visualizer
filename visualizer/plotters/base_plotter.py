from abc import ABC, abstractmethod


class BasePlotter(ABC):
    @abstractmethod
    def plot(self, *args, **kwargs):
        pass
