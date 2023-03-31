from .measurehook import MeasureHook

class MeasureHookLoss(MeasureHook):
    def __init__(self, layer, profiler):
        super().__init__(layer, profiler)
        self.losses = []

    def post_hook(self, module, input, output):
        measurement = self.profiler.end_measurement()
        self.measurements.append(measurement)
        self.losses.append(output.item())

    def get_losses(self):
        return self.losses
    
    def reset_layer_measurements(self):
        self.measurements = []
        self.losses = []


