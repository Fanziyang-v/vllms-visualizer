from abc import ABC, abstractmethod


class IVisualizer(ABC):
    @abstractmethod
    def visualize(self, *args, **kwargs) -> None:
        pass


class IPlotter(ABC):
    @abstractmethod
    def plot(self, *args, **kwargs) -> None:
        pass


class IProcessor(ABC):
    @abstractmethod
    def process(self, *args, **kwargs) -> None:
        pass
