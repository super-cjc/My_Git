import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 生成虚拟数据
def generate_data(nx = 100,nt = 100, x_rabge=(-1,1) ,t_range=(-1,1)):
    x = np.linspace(x_rabge[0],x_rabge[1],nx)       # 生成100个(-1,1)的等间隔数据
    t = np.linspace(t_range[0],t_range[1],nt)

    X,T = np.meshgrid(x,t)                       # 生成对应的n*m网格矩阵
    # 解析解
    c = 1.0
    u_exact = np.sin(2*np.pi*(X - c*T))
    return X.flatten(),T.flatten(),u_exact.flatten() # 返回一维解


# define a simple network
class Net(nn.Module):
    def __init__(self,input_size=2,hidden_size=20,output_size=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.fc4 = nn.Linear(hidden_size,output_size)


    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x


# PINN损失函数
def loss_func(net,x,t,u_exact,c=1.0):
    # 将输入数据进行拼接
    inputs = torch.cat([x,t],dim=1)     # 0: 竖着拼接 1:横着拼接

    # forword
    u_pred = net(inputs)

    # PI部分(其中autograd.grad 用于求高阶导数,create_graph=True 使得计算的结果保留 )
    u_t = torch.autograd.grad(u_pred,t, grad_outputs=torch.ones_like(u_pred),create_graph=True)[0] # du_pred/du_t
    u_x = torch.autograd.grad(u_pred,x, grad_outputs=torch.ones_like(u_pred),create_graph=True)[0]

    # 计算PDE残差
    residual = u_t + c*u_x  # 是因为这里有，当没有损失函数的时候 residual = 0

    # 统计物理损失
    L_PDE = torch.mean(residual ** 2)

    # 数据损失
    L_data = torch.mean((u_exact - u_pred)**2)

    loss = L_PDE + L_data
    return loss

# NN损失函数
def loss_NN_func(net,x,t,u_exact,c=1.0):
    # 将输入数据进行拼接
    inputs = torch.cat([x,t],dim=1)     # 0: 竖着拼接 1:横着拼接

    # forword
    u_pred = net(inputs)

    loss = torch.mean((u_exact - u_pred)**2)

    return loss

def train_PINN():
    # 指定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义输入数据
    X,T,u_exact= generate_data()

    # 转化为张量(requires_grad_(True),保留非叶子节点)
    x = torch.tensor(X,dtype=torch.float32).reshape(-1,1).requires_grad_(True).to(device)
    t = torch.tensor(T,dtype=torch.float32).reshape(-1,1).requires_grad_(True).to(device)
    u_exact = torch.tensor(u_exact,dtype=torch.float32).reshape(-1,1).requires_grad_(True).to(device)

    # 初始化网络
    net = Net().to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)

    # 定义训练轮数
    epoches = 1500
    for epoch in range(epoches):
        optimizer.zero_grad()   # 初始化
        loss = loss_func(net,x,t,u_exact)

        # 反向传播
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{epoches}], Loss: {loss.item()}')

    # 保存模型
    torch.save(net.state_dict(),'PINN_learn.pth')
    return net,x,t,u_exact

def train_NN():
    # 指定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义输入数据
    X,T,u_exact= generate_data()

    # 转化为张量(requires_grad_(True),保留非叶子节点)
    x = torch.tensor(X,dtype=torch.float32).reshape(-1,1).requires_grad_(True).to(device)
    t = torch.tensor(T,dtype=torch.float32).reshape(-1,1).requires_grad_(True).to(device)
    u_exact = torch.tensor(u_exact,dtype=torch.float32).reshape(-1,1).requires_grad_(True).to(device)

    # 初始化网络
    net = Net().to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(),lr=0.001)

    # 定义训练轮数
    epoches = 1500
    for epoch in range(epoches):
        optimizer.zero_grad()   # 初始化
        loss = loss_NN_func(net,x,t,u_exact)

        # 反向传播
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{epoches}], Loss: {loss.item()}')

    # 保存模型
    torch.save(net.state_dict(),'PINN_learn_NN.pth')
    return net,x,t,u_exact



# 可视化结果
def plot_result(net, x, t, u_exact):
    with torch.no_grad():
        # 创建新的规则网格点
        x_plot = torch.linspace(-1, 1, 100, device=x.device).reshape(-1, 1)
        t_plot = torch.full_like(x_plot, 0.5)

        # 组合输入
        inputs = torch.cat([x_plot, t_plot], dim=1)

        # 计算预测值
        u_pred = net(inputs)

        # 计算对应的解析解
        u_exact_plot = torch.sin(2 * np.pi * (x_plot - t_plot))

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot.cpu().numpy(), u_exact_plot.cpu().numpy(), 'b-', label='Exact')
        plt.plot(x_plot.cpu().numpy(), u_pred.cpu().numpy(), 'r--', label='PINN')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title('t = 0.5')
        plt.legend()
        plt.grid(True)
        plt.show()
    pass


if __name__ == '__main__':
    net,x,t,u_exact = train_NN()
    # 加载模型
    model =  Net()
    model.load_state_dict(torch.load('PINN_learn_NN.pth'))
    plot_result(model,x,t,u_exact)


