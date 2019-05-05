def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

class Pipeline(object):
    """
    Defines a pipeline for operating on data. Output of first function will be passed to the second and so forth.

    :param ordered_func_list: (list):
        list of functions to call
    :param func_args: (dict):
        optional dictionary of params to pass to functions in addition to last output
        the dictionary should be in the form of:
        func_name: list(params)
    """

    def __init__(self, ordered_func_list, func_args=None):
        self.pipes = ordered_func_list
        self.func_args = func_args
        self.output = None

    def call(self, input):
        """Apply the functions in current Pipeline to an input.

        :param input: The input to process with the Pipeline.
        """
        out = input
        for pipe in self.pipes:
            if pipe.__name__ in self.func_args:     # if additional arguments present
                all_args = self.func_args[pipe.__name__]
                all_args.insert(0, out)
            else:
                all_args = list(out)
            out = pipe(*all_args)       # pass list to the function to be executed
        return out

    def add_before(self, func, args_dict=None):
        """
        Add a function to be applied before the rest in the pipeline

        :param func: The function to apply
        """
        if args_dict:       # update args dictionary if necessary
            self.func_args = merge_dicts(self.func_args, args_dict)

        self.pipes.insert(0, func)
        return self

    def add_after(self, func, args_dict=None):
        """
        Add a function to be applied at the end of the pipeline

        :param func: The function to apply
        """
        if args_dict:       # update args dictionary if necessary
            self.func_args = merge_dicts(self.func_args, args_dict)

        self.pipes.append(func)
        return self

    @staticmethod
    def identity(x):
        """Return a copy of the input.

        This is here for serialization compatibility with pickle.
        """
        return x
