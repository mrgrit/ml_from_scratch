import numpy as np


# implement softmax function with example 1
def softmax1(a):
    exp_a = np.exp(a)
    sum_exp = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

# implement softmax function with example 2
def softmax2(a):
    exp_a = np.exp(a-c) # to avoid the overflow
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def example(ex_no) :
    if ex_no == 1:
        a = np.array([0.3,2.9,4.0])
        exp_a = np.exp(a)
        print(exp_a)

        sum_exp_a = np.sum(exp_a)
        print(sum_exp_a)

        y = exp_a /sum_exp_a

        print(y)

    if ex_no == 2:
        # avoid overflow
        a = np.array([1010,1000,990])        
        print(np.exp(a) / np.sum(np.exp(a))) #[ nan  nan  nan]

        # so ...
        c = np.max(a)
        print(a - c) #[  0 -10 -20]

        print(np.exp(a-c) / np.sum(np.exp(a-c)))#[  9.99954600e-01   4.53978686e-05   2.06106005e-09]
        
        

example(2)
