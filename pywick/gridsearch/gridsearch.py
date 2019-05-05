import random
import collections

class GridSearch(object):
    """
    Simple GridSearch to apply to a generic function

    :param function: (function):
        function to perform grid search on
    :param grid_params: (dict):
        dictionary mapping variable names to lists of possible inputs aka..\n
        {'input_a':['dog', 'cat', 'stuff'],
        'input_b':[3, 10, 22]}
    :param search_behavior: (string):
        how to perform the search.
        Options are: 'exhaustive', 'sampled_x.x' (where `x.x` is sample threshold 0.0 < 1.0)\n
        `exhaustive` - try every parameter in order they are specified in the dictionary (last key gets all its values searched first)\n
        `sampled`    - sample from the dictionary of params with specified threshold. The random tries *below* the threshold will be executed
    :param args_as_dict: (bool):
        There are two ways to pass parameters into a function:\n
        1. Simply use each key in grid_params as a variable to pass to the function (and change those variable values according
        to the mapping inside grid_params)\n
        2. Pass a single dictionary to the function where the keys of the dictionary themselves are changed according to the
        grid_params\n
        defaults to dict
    """
    def __init__(self, function, grid_params, search_behavior='exhaustive', args_as_dict=True):
        self.func = function
        self.args = grid_params
        self.sampled_thresh = 1.0

        if 'sampled_' in search_behavior:
            behaviors = search_behavior.split('_')
            self.behavior = behaviors[0]
            self.sampled_thresh = float(behaviors[1])
        else:
            self.behavior = search_behavior
        self.args_as_dict = args_as_dict

    def _execute(self, input_args, available_args):
        """
        Recursively reduce parameters and finally execute the function when all params have been selected
        :param input_args:
            dictionary into which to collect input arguments (used in the recursive call to keep just the needed params)
        :param available_args:
            list of available (arg_name, arg_values) tuples for the rest of the arguments
        """

        if len(available_args) == 0:  # We've reached the bottom of the recursive stack, execute function
            doExecute = True
            if self.behavior == 'sampled':
                if random.random() > self.sampled_thresh:
                    doExecute = False

            if doExecute:
                if self.args_as_dict:  # this passes ONE argument to the function which is the dictionary
                    self.func(input_args)
                else:
                    self.func(**input_args)  # this calls the function with arguments specified in the dictionary

        # get all keys
        keys = available_args.keys()
        keys_to_remove = list()

        for i, key in enumerate(keys):
            values = available_args.get(key)

            # this is a list of possible inputs so iterate over it. Strings are iterable in python so filter out
            if isinstance(values, collections.Iterable) and not isinstance(values, str):
                # first, augment available_args so it no longer contains keys that we have already carried over
                keys_to_remove.append(key)
                for k in keys_to_remove:
                    available_args.pop(k)

                for value in values:
                    input_args[key] = value
                    self._execute(input_args, available_args)

                available_args[key] = values  # replace values so they can be used in the next iterative call
                break    # don't do any more iterations after we handled the first key with multiple choices
            else:
                input_args[key] = values
                keys_to_remove.append(key)
                if (i+1) == len(keys):        # we've reached the final item in the available args
                    self._execute(input_args, dict())

    def run(self):
        """
        Runs GridSearch by iterating over options as specified
        :return:
        """

        input_args = dict()
        self._execute(input_args, self.args)
