# functions to help with file io
# standard libraries
import os
import pickle


def write_pickle(obj_to_pickle, file_out_path, file_out_name):

    fpn = os.path.join(file_out_path, file_out_name)

    out_file = open(fpn, 'wb')
    pickle.dump(obj=obj_to_pickle, file=out_file)
    out_file.close()

    return None


def load_pickle(file_in_path, file_in_name):

    fpn = os.path.join(file_in_path, file_in_name)
    in_file = open(fpn, 'rb')
    unpickled = pickle.load(in_file)
    in_file.close()

    return unpickled


if __name__ == '__main__':

    example_dict = {1: "6", 2: "2", 3: "f"}

    f_path = 'H:/temp'
    f_name = 'that_file.pkl'
    write_pickle(example_dict, f_path, f_name)

    testo = load_pickle(f_path, f_name)

    example_dict['4'] = 'bob'
    print(testo)
    print(example_dict)
