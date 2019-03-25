import json
from .gridsearch import GridSearch

my_args = {
    'shape': '+plus+',
    'animal':['cat', 'mouse', 'dog'],
    'number':[4, 5, 6],
    'device':['CUP', 'MUG', 'TPOT'],
    'flower' : '=Rose='
}

def tryme(args_dict):
    print(json.dumps(args_dict, indent=4))

def tryme_vars(animal='', number=0, device='', shape='', flower=''):
    print(animal + " : " + str(number) + " : " + device + " : " + shape + " : " + flower)

def main():
    grids = GridSearch(tryme, grid_params=my_args, search_behavior='exhaustive', args_as_dict=True)

    print('-------- INITIAL SETTINGS ---------')
    tryme(my_args)
    print('-------------- END ----------------')
    print('-------------- --- ----------------')
    print()
    print('+++++++++++ Dict Result ++++++++++')
    grids.run()
    print('+++++++++++++ End Dict Result +++++++++++\n\n')

    grids = GridSearch(tryme_vars, grid_params=my_args, search_behavior='sampled_0.5', args_as_dict=False)
    print('========== Vars Result ==========')
    grids.run()
    print('========== End Vars Result ==========')
    # exit()

if __name__ == '__main__':
    main()
