# GAtorch

Green AI torch tries to create awereness over energy consumption within the Pytorch ML framework. Its goal is to measure energy consumption throughout the complete process of AI engineers and can give overviews and indications for performance gains with respect to energy consumption. 

Currently it supports energy measurement of the training passes per layer. 

# Installation requirements
- cuda >=11.7
- cuDNN >=8

Create a virtualenv like `virtualenv .venv` and activate it using `source ./.venv/bin/activate` and install the other requirements with `pip install requirements.txt`.

## Basic example
```python
from GA import GA

# Create the profiler object and attach a model to it
ga_measure = GA()
ga_measure.attach_model(model)

# Let's try to do a single forward pass and a backward pass
x = torch.zeros([1, 1, 28, 28]).to(device)
y = torch.zeros([1, 10]).to(device)
pred = model(x) # forward

loss = loss_fn(pred, y)
optimizer.zero_grad()
loss.backward() # backward
optimizer.step()

# Now lets print the mean measurements
print(ga_measure.get_mean_measurements())
```

or run the example scripts in the `examples` directory.

## Compatability

Some older hardware might not support energy consumption measurements:
- NVML requires Tesla Architecture NVIDIA GPUs or newer to work.
- RAPLs DRAM measurements are only available for XENON CPUs. 

In case you get compatibility errors due to older hardware you can disable the failing measurement application, use the `disable_measurements` parameter in the `attach_model` function. This parameter accepts a list of disabled measurements out of `['cpu', 'ram', 'gpu']`, default is `[]`. You need to use at least one measurement that is not disabled. The program will indicate that the disabled devices are unavailable. 

## Permissions
<!-- In case you get a permission error, run:
```bash
sudo chmod -R a+r /sys/class/powercap/intel-rapl
``` 
DONT DO THIS, THIS IS A SECURITY VULNERABILITY
-->
Due to [Platypus attack](https://platypusattack.com) Intel RAPL requires root permission for energy readings. In order to run this program with the correct permissions, do NOT make Intel RAPL readable for any user as this introduces vulernability. Instead use Python with sudo instead:
```bash
sudo ./.venv/bin/python <script_name>.py
```

## Tensorboard
This tool can automatically generate energy consumption reports and display these in Tensorboard. To use Tensorboard run `tensorboard --logdir=runs` and open the browser to view the graphs. This tool further allows for custom graph generation and tensorboard integration, but is not complete and needs to be extended.  

# Roadmap

The current architecture of this tool uses the integrated hooks of the PyTorch library, which restricts the current implementation towards the final goal of complete coverage including data loading, pre-processing, saving and loading a model etc. To give a more thorough analysis of the imapct of energy consumption in ML development, this still needs to be developed.

This tool differes from other tools by measuring in-depth layers and system componenents and could be expanded to provide energy consumption data that can lead to recommendations for eliminating certain layers due to high energy consumption compared to accuracy gain. 

PyJoules measures the energy consumption per individual hardware components and this data could be sereperated in order to provide a relative component view. Another improvement could be to measure the system component utilisation over time, which can be an indicator of wasted energy. 
