from abc import ABC, abstractmethod

class EnergyProfiler(ABC):

    @abstractmethod
    def start_measurement(self):
        pass

    @abstractmethod
    def end_measurement(self):
        return None
