import numpy as np




def sigmod(x):
    return 1/1+np.exp(-x)


if __name__ == '__main__':
    input_list = [1,2,3,4,5,6,7,8,9]
    input_list_T = np.array(input_list,ndmin = 2).T
    input_list = sigmod(input_list_T)
    print(input_list)