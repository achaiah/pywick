"""
Conditions are useful for any custom pre- and post-processing that must be done on batches of data.
Module trainer maintains two separate condition lists that are executed before/after the network forward pass.

An example of a condition could be an Assert that needs to be performed before data is processed.
A more advanced example of a condition could be code that modifies the network based on input or output
"""

from enum import Enum, auto
from .misc import is_tuple_or_list

class CondType(Enum):
    PRE = auto()
    POST = auto()

class ConditionsContainer(object):
    '''
    This container maintains metadata about the execution environment in which the conditions are performed

    exec_type of the container indicates whether it is being run during training or evaluation
    '''
    def __init__(self, exec_type, prefix=''):
        '''
        :param exec_type: ExecType of the container (metadata flag about its execution environment)
        :param prefix: Custom prefix (if any) for output logs
        '''
        self.conditions = {CondType.PRE:[], CondType.POST:[]}
        self.prefix = prefix
        self.exec_type = exec_type

    def add_preconditions(self, conditions):
        '''
        :param conditions: pre-condition(s) to add - can be single or a list
        '''
        self._add_conditions(conditions, CondType.PRE)

    def add_postconditions(self, conditions):
        '''
        :param conditions: post-condition(s) to add - can be single or a list
        '''
        self._add_conditions(conditions, CondType.POST)

    def _add_conditions(self, conditions, type):
        '''
        :param conditions: condition(s) to add - can be single or a list
        :param type: CondType
        :return:
        '''
        conditionz = [conditions] if not is_tuple_or_list(conditions) else conditions
        self.conditions[type].extend(conditionz)


    def reset(self):
        '''
        Reset conditions in the container
        :return:
        '''
        for condition in self.conditions[CondType.PRE]:
            condition.reset()
        for condition in self.conditions[CondType.POST]:
            condition.reset()


    def __call__(self, cond_type, epoch_num, batch_num, net=None, input_batch=None, output_batch=None, target_batch=None):
        '''

        :param cond_type: ContType to execute
        :param epoch_num: Number of the current epoch
        :param batch_num: Number of the current batch
        :param net: Network that is being used
        :param input_batch: Input that is being used
        :param output_batch: Output that was generated in the forward pass
        :param target_batch: Ground truth if available
        :return:
        '''
        logs = {}
        for condition in self.conditions[cond_type]:
            logs_out = condition(self.exec_type, epoch_num, batch_num, net, input_batch, output_batch, target_batch)
            if logs_out is not None:
                logs[self.prefix + condition._name] = logs_out
        return logs

class Condition(object):
    """
    Default class from which all other Condition implementations inherit.
    """

    def __call__(self, exec_type, epoch_num, batch_num, net=None, inputs=None, outputs=None, labels=None):
        '''
        :param exec_type: Type of execution from ExecType enum
        :param epoch_num: The epoch of execution
        :param batch_num: The batch of execution
        :param net: network which did the forward pass
        :param inputs: The inputs that were used
        :param outputs: Outputs of the forward pass
        :param labels: Ground Truth

        :return:
        '''
        raise NotImplementedError('Custom Conditions must implement this function')

    def reset(self):
        raise NotImplementedError('Custom Conditions must implement this function')


# class ConditionCallback(Callback):
#
#     def __init__(self, container):
#         self.container = container
#     def on_epoch_begin(self, epoch_idx, logs):
#         self.container.reset()



class SegmentationInputAsserts(Condition):
    '''
    Executes segmentation-specific asserts before executing forward pass on inputs
    '''

    def __call__(self, exec_type, epoch_num, batch_num, net=None, inputs=None, outputs=None, labels=None):
        assert inputs.size()[2:] == labels.size()[1:]

    def reset(self):
        pass


class SegmentationOutputAsserts(Condition):
    '''
    Executes segmentation-specific asserts after executing forward pass on inputs
    '''

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, exec_type, epoch_num, batch_num, net=None, inputs=None, outputs=None, labels=None):
        if isinstance(outputs, tuple):  # if we have an auxiliary output as well
            if any(item is None for item in outputs) or len(outputs) < 2:      # seriously... why?  I'm looking at you OCNet
                outs = outputs[0]
            else:
                outs, aux = outputs
        else:
            outs = outputs
        assert outs.size()[2:] == labels.size()[1:]
        assert outs.size()[1] == self.num_classes

    def reset(self):
        pass
