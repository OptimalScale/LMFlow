# https://pypi.org/project/names-dataset/
# pip install names-dataset==3.1.0
# python create_dataset_for_symbolic_reasoning.py --dataset=coin_flip --dataset_size=500
# python create_dataset_for_symbolic_reasoning.py --dataset=last_letters --dataset_size=500
from names_dataset import NameDataset
from collections import OrderedDict
import random
import argparse
import json

def create_dataset(args):
    # Get name list from NameDataset library ...
    nd = NameDataset()
    n_name_samples = int(args.dataset_size * args.names_in_sample / 2) # Males and Females
    name_list = nd.get_top_names(n=n_name_samples, country_alpha2='US')
    #print(name_list)
    name_list = name_list['US']['M'] + name_list['US']['F']
    #print(name_list)
    random.shuffle(name_list)
    #print(len(name_list))
    #print(name_list)

    # Create samples one by one ...
    sample_list = []
    for i in range(args.dataset_size):
        if args.dataset == 'coin_flip':
            q = 'A coin is heads up. '
            a = 1
            for j in range(args.names_in_sample):
                flip_result = random.randint(0, 1)
                a += flip_result
                k = i*args.names_in_sample + j
                text = "flips" if flip_result == 1 else "does not flip"
                q += '{} {} the coin. '.format(name_list[k], text)
            q += 'Is the coin still heads up? Note that "flip" here means "reverse".'
            a = 'yes' if (a % 2) == 1 else 'no'
        elif args.dataset == 'last_letters':
            q = 'Take the last letters of each words in "'
            a = ''
            for j in range(args.names_in_sample):
                k = i*args.names_in_sample + j
                q += name_list[k]
                if j != (args.names_in_sample-1):
                    q += ' '
                a += name_list[k][-1]
            q += '" and concatenate them.'
        else:
            raise ValueError("dataset is not properly defined ...")
        # store created sample into arrays ...
        #print(q)
        #print(a)
        dic = OrderedDict()
        dic["question"] = q
        dic["answer"] = a
        sample_list.append(dic)
        
    # save data into file as json format ...
    json_data =  OrderedDict()
    json_data["examples"] = sample_list
    #print(json.dumps(json_data, indent=4))
    with open(args.dataset + '.json', 'w') as f:
        json.dump(json_data, f, indent=4)
    
def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    random.seed(args.random_seed)
    create_dataset(args)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Creat Dataset For Symbolic Task Reasoning")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument("--dataset", type=str, default="coin_flip", choices=["coin_flip", "last_letters"], help="")
    parser.add_argument("--dataset_size", type=int, default=10, help="")
    parser.add_argument("--names_in_sample", type=int, default=4, help="")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()