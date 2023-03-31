from torch.utils.tensorboard import SummaryWriter
from codecarbon import EmissionsTracker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from .profilers import ProfilerPyJoules, MeasureHook, MeasureHookLoss

class GA:
    def __init__(self):
        self.hook_register = {}
        self.hook_handles = []
        self.named_layer = False
        self.model = None
        self.loss = None
        self.emission_tracker = None
    
    def attach_model(self, model, loss=None, named_layer=True, profiler='pyjoules', disable_measurements=[]):
        '''
        Attach a model to the GA profiler. You can add also a loss function. 
        The profiler will then track the energy consumption of the attached model.

        :param model: PyTorch model you want to track.
        :param loss: Loss function to be passed for tracking the consumption of the loss computation.
        :param named_layers: Set to False to not track individual named layers.
        :param profiler: Typed of profiler used to track the energy consumption. Currently only 'pyjoules' is implemented.
        :param disable_measurements: Set which hardware components you are not intrested in tracking between cpu, ram and gpu.
        :type model: torch.nn.module
        :type loss: torch.nn.module
        :type named_layers: boolean
        :type profiler: string
        :type disable_measurements: list
        '''
        if self.model is not None:
            raise RuntimeError('Already a model attached, use detach_model() before attaching the profiler to a new one.')

        self.model = model

        device = next(model.parameters()).device
        device_type = device.type
        device_idx = device.index if device.index else 0

        if named_layer:
            self.named_layer = True
            for name, layer in model.named_children():
                self._register_layer_hook(name, layer, profiler, device_type, device_idx, disable_measurements)

        self._register_layer_hook(model.__class__.__name__, model, profiler, device_type, device_idx, disable_measurements)
        
        if loss is not None:
            self.loss = loss
            self._register_layer_hook('loss', self.loss, profiler, device_type, device_idx, disable_measurements)
        

    def _register_layer_hook(self, name, layer, profiler, device_type, device_idx, disable_measurements):
        if profiler=='pyjoules':
            profiler = ProfilerPyJoules(device=device_type, index=device_idx, disable=disable_measurements)

        if name=='loss':
            forward_hook = MeasureHookLoss(name, profiler)
            backward_hook = MeasureHook(name, profiler)    
        else:
            forward_hook = MeasureHook(name, profiler)
            backward_hook = MeasureHook(name, profiler)

        self.hook_register[name] = (forward_hook, backward_hook)

        handle_pre_fw = layer.register_forward_pre_hook(forward_hook.pre_hook)
        handle_post_fw = layer.register_forward_hook(forward_hook.post_hook)

        handle_pre_bw = layer.register_full_backward_pre_hook(backward_hook.pre_hook)
        handle_post_bw = layer.register_full_backward_hook(backward_hook.post_hook)

        self.hook_handles.extend([handle_pre_fw, handle_post_fw, handle_pre_bw, handle_post_bw])

    def _check_model_attached(self):
        if self.model is None:
            raise RuntimeError('No model attached to profiler.')
        
    def detach_model(self):
        '''
        Detach the current model from the GA profiler.
        '''
        self._check_model_attached()

        for handle in self.hook_handles:
            handle.remove()
        self.hook_register = {}
        self.hook_handles = []
        self.model = None

    def get_full_measurements(self):
        '''
        Get the full measurements collected by the profiler.

        :returns: The energy consumptions for each pass and model component.
        :rtype: dict
        '''
        self._check_model_attached()

        measurements = {}
        for name, (forward_hook, backward_hook) in self.hook_register.items():
            ga_fw = forward_hook.get_layer_measurements()
            ga_bw = backward_hook.get_layer_measurements()
            measurements[name] = {'forward':ga_fw, 'backward':ga_bw}

        return measurements

    def get_mean_measurements(self):
        '''
        Get the mean of the measurements collected by the profiler.

        :returns: The mean energy consumption for each model component.
        :rtype: dict
        '''
        self._check_model_attached()

        measurements = self.get_full_measurements()
        for _, ly_dict in measurements.items():
            ly_dict['forward'] = sum(ly_dict['forward'])/len(ly_dict['forward']) if len(ly_dict['forward'])>0 else 0
            ly_dict['backward'] = sum(ly_dict['backward'])/len(ly_dict['backward']) if len(ly_dict['backward'])>0 else 0
        return measurements
    
    def get_sum_measurements(self):
        '''
        Get the sum of the measurements collected by the profiler.

        :returns: The full energy consumption for each model component.
        :rtype: dict
        '''
        self._check_model_attached()

        measurements = self.get_full_measurements()
        for _, ly_dict in measurements.items():
            ly_dict['forward'] = sum(ly_dict['forward']) if len(ly_dict['forward'])>0 else 0
            ly_dict['backward'] = sum(ly_dict['backward']) if len(ly_dict['backward'])>0 else 0
        return measurements
    
    def get_losses(self):
        self._check_model_attached()
        if self.loss is None:
            raise RuntimeError('Loss has not been attached to the profiler.')
        
        return self.hook_register['loss'][0].get_losses()
    
    def to_pandas(self):
        '''
        Convert the energy measurements into a pandas.DataFrame object.

        :returns: The dataframe of the energy consumption for each model component.
        :rtype: pandas.DataFrame
        '''
        measurements = self.get_full_measurements()
        pd_dict = {}
        for column in measurements.keys():
            pd_dict[f'{column}_forward'] = measurements[column]['forward']
            pd_dict[f'{column}_backward'] = measurements[column]['backward']
        
        return pd.DataFrame.from_dict(pd_dict, orient='index').T

    def to_csv(self, filename):
        '''
        Save the energy measurements into a csv file.

        :param filename: Name of the csv file.
        :type filename: string
        '''
        if filename[-4:]!='.csv':
            filename = filename + '.csv'

        df = self.to_pandas()
        df.to_csv(filename)

    def visualize_data(self, layers='all', complete_model=True, loss=False, phase='total', kind='line', smoothing=0.3, figsize=None, filename=None):
        '''
        Generate a matplotlib plot for the energy measurements.

        :param layers: Pass in a list which named layers you want to display. Pass 'all' if you want to see all of them.
        :param complete_model: Set to False if you don't want to see the data for the complete model.
        :param loss: Set to True to display also the loss function data.
        :param phase: Select which phase to display between 'total', 'forward' or 'backward'.
        :param kind: Select which type of plot to generate between 'line', 'violin' or 'box'.
        :param smoothing: Only used for lineplots. Value between 0 and 1 used to add smoothing to the displayed data.
        :param figsize: Size of the figure generated.
        :param filename: Pass a filename if you want to save the image created.
        :type layers: list or string
        :type complete_model: boolean
        :type loss: boolean
        :type phase: string
        :type kind: string
        :type smoothing: float
        :type figsize: Tuple
        :type filename: string
        :returns: The matplotlib axes containing the plot.
        :rtype: Axes
        '''    
        df = self.to_pandas()
        df = df.dropna(axis=1)

        if not complete_model:
            modelname = self.model.__class__.__name__
            if f'{modelname}_forward' in df.columns:
                df.drop(labels=[f'{modelname}_forward'], axis=1, inplace=True)
            if f'{modelname}_backward' in df.columns:
                df.drop(labels=[f'{modelname}_backward'], axis=1, inplace=True)

        if not loss:
            if f'loss_forward' in df.columns:
                df.drop(labels=[f'loss_forward'], axis=1, inplace=True)
            if f'loss_backward' in df.columns:
                df.drop(labels=[f'loss_backward'], axis=1, inplace=True)

        if layers!='all':
            for column, _ in self.model.named_children():
                if f'{column}' not in layers and f'{column}_forward' in df.columns:
                    df.drop(labels=[f'{column}_forward'], axis=1, inplace=True)
                if f'{column}' not in layers and f'{column}_backward' in df.columns:
                    df.drop(labels=[f'{column}_backward'], axis=1, inplace=True)
                
        if phase=='total':
            for layer, _ in self.model.named_children():
                if (layers=='all' or (layer in layers)) and self.named_layer:
                    df[layer]=[0.0 for _ in range(len(df.index))]
                    if f'{layer}_forward' in df.columns:
                        df[layer] = df[layer] + df[f'{layer}_forward']
                        df.drop(labels=[f'{layer}_forward'], axis=1, inplace=True)
                    if f'{layer}_backward' in df.columns:
                        df[layer] = df[layer] + df[f'{layer}_backward']
                        df.drop(labels=[f'{layer}_backward'], axis=1, inplace=True)
            if complete_model:
                modelname = self.model.__class__.__name__
                df[modelname]=[0.0 for _ in range(len(df.index))]
                if f'{modelname}_forward' in df.columns:
                    df[modelname] = df[modelname] + df[f'{modelname}_forward']
                    df.drop(labels=[f'{modelname}_forward'], axis=1, inplace=True)
                if f'{modelname}_backward' in df.columns:
                    df[modelname] = df[modelname] + df[f'{modelname}_backward']
                    df.drop(labels=[f'{modelname}_backward'], axis=1, inplace=True)
            if loss and self.loss is not None:
                df['loss']=[0.0 for _ in range(len(df.index))]
                if f'loss_forward' in df.columns:
                    df['loss'] = df['loss'] + df[f'loss_forward']
                    df.drop(labels=[f'loss_forward'], axis=1, inplace=True)
                if f'loss_backward' in df.columns:
                    df['loss'] = df['loss'] + df[f'loss_backward']
                    df.drop(labels=[f'loss_backward'], axis=1, inplace=True)
        elif phase=='forward':
            columns = df.columns
            for column in columns:
                if column.endswith('_backward'):
                    df.drop(labels=[column], axis=1, inplace=True) 
        elif phase=='backward':
            columns = df.columns
            for column in columns:
                if column.endswith('_forward'):
                    df.drop(labels=[column], axis=1, inplace=True) 

        if len(df.index)==0:
            return None 
        
        if figsize is not None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()
        
        if kind=='line':
            if smoothing>0:
                if smoothing>1:
                    raise ValueError('smoothing can only be between 0 and 1.')
                for column in df:
                    rows_count = len(df.index)
                    window_length = ((rows_count-4)*smoothing)+4
                    df[column]=signal.savgol_filter(df[column], window_length, 3, mode="nearest")

            df.plot(xlabel='Iteration', ylabel='Energy (J)', ax=ax)

        elif kind=='box':
            df.plot(kind='box', ax=ax)
            ax.set_ylabel('Energy (J)')

        elif kind=='violin':
            ax.violinplot([df[column].tolist() for column in df])
            ax.set_xticks([x for x in range(1,len(df.columns)+1)]) 
            ax.set_xticklabels(df.columns)
            ax.set_ylabel('Energy (J)')

        else:
            raise ValueError('kind parameter can only be line, box or violin.')

        fig.tight_layout()

        if filename:
            fig.savefig(filename)

        return ax
        
    def start_tracker_emissions(self, save_to_file=False):
        '''
        Start a tracker for carbon emission. Implemented using codecarbon.

        :param save_to_file: Save the tracking results in a file called 'emissions'.
        :type save_to_file: boolean
        '''
        if self.emission_tracker is not None:
            print(self.emission_tracker)
            raise RuntimeError('Emission tracker already running.')
        
        self.emission_tracker = EmissionsTracker(save_to_file=save_to_file)
        self.emission_tracker.start()
    
    def stop_tracker_emissions(self):
        '''
        Stop the tracker for carbon emission. 

        :returns: The total emissions produced by the model since the tracker was started.
        :rtype: float
        '''
        if self.emission_tracker is None:
            raise RuntimeError('Emission tracker is not running. Use start_tracker_emissions().')
        
        emissions: float = self.emission_tracker.stop()
        self.emission_tracker = None
        return emissions
    
    def set_tensorboard_stats(self, writer=None, experiment=0, named_layer=True, sample_size=500):
        '''
        Sets the tensorboard data that can be viewed through the tensorboard dashboard. 

        :param writer: Optionally pass an already existing tensorboard writer.
        :param experiment: Set an identifier for the displayed data.
        :param named_layer: Indicate if the named layers should also be displayed in tensorboard.
        :param sample_size: Size of the sample size used to define a loss step. Used to compute the grap for energy per loss step.
        :type writer: torch.utils.tensorboard.SummaryWriter
        :type experiment: int
        :type named_layer: boolean
        :type sample_size: int 
        '''
        if writer is None:
            writer = SummaryWriter(f'runs/experiment_{experiment}')
        
        df = self.to_pandas()
        df = df.dropna(axis=1)

        if named_layer and self.named_layer:
            for layer, _ in self.model.named_children():
                df[layer]=[0.0 for _ in range(len(df.index))]
                if f'{layer}_forward' in df.columns:
                    df[layer] = df[layer] + df[f'{layer}_forward']
                if f'{layer}_backward' in df.columns:
                    df[layer] = df[layer] + df[f'{layer}_backward']
        else:
            for column, _ in self.model.named_children():
                if f'{column}_forward' in df.columns:
                    df.drop(labels=[f'{column}_forward'], axis=1, inplace=True)
                if f'{column}_backward' in df.columns:
                    df.drop(labels=[f'{column}_backward'], axis=1, inplace=True)

        modelname = self.model.__class__.__name__
        df[modelname]=[0.0 for _ in range(len(df.index))]
        if f'{modelname}_forward' in df.columns:
            df[modelname] = df[modelname] + df[f'{modelname}_forward']
        if f'{modelname}_backward' in df.columns:
            df[modelname] = df[modelname] + df[f'{modelname}_backward']

        if self.loss is not None: 
            df['loss']=[0.0 for _ in range(len(df.index))]
            if f'loss_forward' in df.columns:
                df['loss'] = df['loss'] + df[f'loss_forward']
            if f'loss_backward' in df.columns:
                df['loss'] = df['loss'] + df[f'loss_backward']

        if len(df.index)==0:
            return None 
        
        fw_columns = [column for column in df.columns if column.endswith('_forward')]
        bw_columns = [column for column in df.columns if column.endswith('_backward')]
        t_columns = [column for column in df.columns if not column.endswith('_forward') and not column.endswith('_backward')]

        loss_column = np.array(self.get_losses())
        loss_mean = np.mean(loss_column[:(len(loss_column)//sample_size)*sample_size].reshape(-1, sample_size), axis=1)
        loss_mean_diff = -1*np.diff(loss_mean)

        energy_column = df.iloc[:]['NeuralNetwork'].to_numpy()
        energy_sum = np.sum(energy_column[:(len(energy_column)//sample_size)*sample_size].reshape(-1, sample_size), axis=1)[1:]
        energy_loss_ratio = energy_sum / loss_mean_diff
        energy_loss_ratio[energy_loss_ratio < 0] = 0

        for i in range(len(energy_loss_ratio)):
            writer.add_scalar('Energy per 1 Unit Loss Decrease', energy_loss_ratio[i], i)
            writer.add_scalar('Average Loss per Epoch', loss_mean[i], i)

        for i in df.index:
            writer.add_scalars('Energy Consumption', dict([(key, df.iloc[i][key].item()) for key in t_columns]), i)
            writer.add_scalars('Forward Pass', dict([(key, df.iloc[i][key].item()) for key in fw_columns]), i)
            if len(bw_columns)>0:
                writer.add_scalars('Backward Pass', dict([(key, df.iloc[i][key].item()) for key in bw_columns]), i)

        for layer in t_columns:
            writer.add_text(f'Total energy consumption: {layer}', f'{round(df[layer].sum(), 2)} J')
    
        writer.close()
        
    def reset(self, emission_tracker=True):
        '''
        Reset all the measurements. 

        :param emission_tracker: Indicates if the emissions tracker needs to be reset as well.
        :type emission_tracker: boolean
        '''
        self._check_model_attached()

        for _, (forward_hook, backward_hook) in self.hook_register.items():
            forward_hook.reset_layer_measurements()
            backward_hook.reset_layer_measurements()

        if emission_tracker and self.emission_tracker is not None:
            self.stop_tracker_emissions()
