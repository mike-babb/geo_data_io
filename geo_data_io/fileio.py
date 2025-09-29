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


def write_json(json_data: str, output_file_path: str,
               output_file_name: str, var_name: str = None):

    ofpn = os.path.join(output_file_path, output_file_name)
    if var_name is None:
        var_name_str = os.path.splitext(output_file_name)[0]
    else:
        var_name_str = var_name

    print(var_name_str)
    with open(ofpn, 'w') as file:
        write_line = 'var {} ='.format(var_name_str)
        file.write(write_line)
        file.write(json_data)

    return None


if __name__ == '__main__':

    example_dict = {1: "6", 2: "2", 3: "f"}

    f_path = 'H:/temp'
    f_name = 'that_file.pkl'
    write_pickle(example_dict, f_path, f_name)

    testo = load_pickle(f_path, f_name)

    example_dict['4'] = 'bob'
    print(testo)
    print(example_dict)
