class MeasureHook:
    def __init__(self, layer, profiler):
        self.layer = layer 
        self.measurements = []
        self.profiler = profiler

    def pre_hook(self, *args):
        self.profiler.start_measurement()

    def post_hook(self, *args):
        measurement = self.profiler.end_measurement()
        self.measurements.append(measurement)

    def get_layer_measurements(self):
        return self.measurements
    
    def reset_layer_measurements(self):
        self.measurements = []

