"""
三层神经网络
"""
import numpy as np



class Netural_network_simple():
    """
    简单的三层神经网络
    """
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learn_nate):
        """
        神经网络初始化
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learn_nate = learn_nate
        self.wih = np.random.normal(0.0,pow(self.hidden_nodes,-0.5),(self.hidden_nodes,self.input_nodes))
        self.who = np.random.normal(0.0,pow(self.output_nodes,-0.5),(self.output_nodes,self.hidden_nodes))

        pass

    #激活函数
    def sigmod(self,x):
        return 1 / (1 + np.exp(-x))


    def train(self,input_list,target_list):
        """
        神经网络训练
        """
        inputs = np.array(input_list,ndmin=2).T
        targets = np.array(target_list,ndmin=2).T
        #计算隐藏层的输入输出
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.sigmod(hidden_inputs)
        #计算输出层的输入输出
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.sigmod(final_inputs)
        #BP算法进行误差反馈，
        # 跟新who——隐藏层和输出层之间的权重
        # 跟新wih——输入层和隐藏层之间的权重
        error_final = targets - final_outputs
        self.who += self.learn_nate * np.dot((error_final*final_outputs*(1-final_outputs))
                                             ,np.transpose(hidden_outputs))
        #!!!得出隐藏层的输出误差
        hidden_error = np.dot(self.who.T,error_final)
        self.wih += self.learn_nate * np.dot((hidden_error*hidden_outputs*(1-hidden_outputs))
                                             ,np.transpose(inputs))


        pass
    def quary(self,input_list,target_list):
        """
        查询神经网络
        """
        inputs = np.array(input_list,ndmin=2).T
        #计算隐藏层的输入输出
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.sigmod(hidden_inputs)
        #计算输出层的输入输出
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.sigmod(final_inputs)

        return final_outputs


