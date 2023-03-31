from pyJoules.device import DeviceFactory
from pyJoules.device.rapl_device import RaplPackageDomain, RaplDramDomain, RaplDevice
from pyJoules.device.nvidia_device import NvidiaGPUDomain, NvidiaGPUDevice
from pyJoules.energy_meter import EnergyMeter
from pyJoules.exception import NoSuchDeviceError
from .energyprofiler import EnergyProfiler

class ProfilerPyJoules(EnergyProfiler):

    def __init__(self, device='cpu', index=0, disable=[]):
        self.sample_keys = []
        self.disable = disable
        domains = self.get_domains(device, index)
        devices = DeviceFactory.create_devices(domains)
        self.meter = EnergyMeter(devices)

    def get_available_devices(self):
        try:
            all_available = RaplDevice.available_domains()
        except NoSuchDeviceError:
            all_available = []
        all_available = list(map(lambda x: x.get_domain_name(), all_available))
        try:
            nvidia_available = NvidiaGPUDevice.available_domains()
        except NoSuchDeviceError:
            nvidia_available = []
        all_available += list(map(lambda x: x._repr.replace("_0", ""), nvidia_available))
        return all_available

    def check_available_devices(self, device):
        all_available = self.get_available_devices()
        selected = []
        unavailable = []
        if device == "cpu":
            selected.append("package") if "package" in all_available and not "cpu" in self.disable else unavailable.append("package")
            selected.append("dram") if "dram" in all_available and not "ram" in self.disable else unavailable.append("dram")
        if device == "cuda":
            selected.append("package") if "package" in all_available and not "cpu" in self.disable else unavailable.append("package")
            selected.append("dram") if "dram" in all_available and not "ram" in self.disable else unavailable.append("dram")
            selected.append("nvidia_gpu") if "nvidia_gpu" in all_available and not "gpu" in self.disable else unavailable.append("nvidia_gpu")
        return (selected, unavailable)

    def get_domains(self, device, index):
        selected, unavailable = self.check_available_devices(device)
        if len(unavailable) > 0:
            print(f'Unavailable devices for {self.__class__}: {unavailable}.')

        if len(selected) == 0:
            raise RuntimeError(f'No available devices for {self.__class__}')
        
        domains = []
        for s in selected:
            domains.append(RaplPackageDomain(index)) if s == "package" else None
            domains.append(RaplDramDomain(index)) if s == "dram" else None
            domains.append(NvidiaGPUDomain(index)) if s == "nvidia_gpu" else None
            self.sample_keys.append(f'{s}_{index}')
        return domains

    def start_measurement(self):
        self.meter.start()

    def end_measurement(self):
        self.meter.stop()
        return self.get_measurement()

    def get_measurement(self):
        measurement = 0
        for sample in self.meter.get_trace()._samples:
            for key in self.sample_keys:
                if 'nvidia_gpu' in key: # Convert from mJ (nvml) to J
                    measurement += sample.energy[key] / 1000
                else: # Convert from uJ (rapl) to J
                    measurement += sample.energy[key] / 1000 / 1000
        return measurement
