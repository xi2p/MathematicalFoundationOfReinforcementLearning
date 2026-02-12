import sys
sys.path.append('..')

from mforl.function.grid_world_model import GridWorldModel, PolicyTabular
from mforl.function.basic import State, Action, Reward
import torch
import random
from copy import deepcopy


# model
grid = GridWorldModel(
    width=3,
    height=3,
    forbidden_states=[State((0, 1)), State((0, 2)), State((2, 1))],
    terminal_states=[State((2, 2))]
)

# action
action_up = grid.ACTION_UP
action_down = grid.ACTION_DOWN
action_left = grid.ACTION_LEFT
action_right = grid.ACTION_RIGHT
action_stay = grid.ACTION_STAY

policy = PolicyTabular(grid.states, grid.actions)
policy[action_right | State((0, 0))] = torch.tensor(1.0)
policy[action_down | State((1, 0))] = torch.tensor(1.0)
policy[action_left | State((2, 0))] = torch.tensor(1.0)
policy[action_right | State((0, 1))] = torch.tensor(1.0)
policy[action_down | State((1, 1))] = torch.tensor(1.0)
policy[action_left | State((2, 1))] = torch.tensor(1.0)
policy[action_right | State((0, 2))] = torch.tensor(1.0)
policy[action_right | State((1, 2))] = torch.tensor(1.0)
policy[action_stay | State((2, 2))] = torch.tensor(1.0)


print(grid)

# Define the state value function
# Use polygonal linear function approximation
order = 10   # Take 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3 as features
# parameter
w = torch.tensor([ 6.7202,  1.7982,  2.1958,  0.8145,  0.4604,  0.8455,  0.3227,  0.1086,
         0.1854,  0.1704,  0.0768, -0.0674,  0.0744,  0.0478, -0.1673, -0.0461,
        -0.1554,  0.0189,  0.0573, -0.0210, -0.3361, -0.1076, -0.1994, -0.0089,
         0.0620,  0.0487, -0.0553, -0.4205, -0.1384, -0.2214, -0.0227,  0.0644,
         0.0836,  0.0445, -0.0725, -0.4627, -0.1537, -0.2324, -0.0297,  0.0656,
         0.1010,  0.0944,  0.0423, -0.0811, -0.4838, -0.1614, -0.2379, -0.0332,
         0.0662,  0.1097,  0.1193,  0.0998,  0.0413, -0.0854, -0.4944, -0.1653,
        -0.2406, -0.0349,  0.0665,  0.1141,  0.1318,  0.1285,  0.1025,  0.0407,
        -0.0876, -0.4997])

def feature_extractor(state: State) -> torch.Tensor:
    x, y = state.value
    # normalize x, y to [0, 1]
    x = x / (grid.width - 1)
    y = y / (grid.height - 1)

    # features = torch.tensor([
    #     1.0,
    #     x,
    #     y,
    #     x**2,
    #     x*y,
    #     y**2,
    #     x**3,
    #     x**2 * y,
    #     x * y**2,
    #     y**3
    # ], dtype=torch.float32)

    # Polygonal feature
    tensor = []
    for o in range(0, order+1):
        for i in range(o+1):
            tensor.append(x**i * y**(o-i))

    # Fourier basis features
    # cos(c1 x + c2 y)
    # tensor = []
    # for i in range(order + 1):
    #     for j in range(order + 1):
    #         tensor.append(torch.cos(torch.tensor((i * x + j * y) * 3.1415926)))
    features = torch.tensor(tensor, dtype=torch.float32)

    return features

def net(input: torch.Tensor) -> torch.Tensor:
    return torch.dot(w, input)



# Print learned state values
for state in grid.states:
    state_value = net(feature_extractor(state)).item()
    print(f"State: {state}, Learned Value: {state_value:.4f}")

print("Learned parameters w:", w)



# tensor([ 6.7403, -0.7160,  1.5544, -0.7160,  0.0327, -1.0470, -0.1502,  1.2027,
#          0.0308, -0.1758, -0.6556, -0.1758,  0.0327,  1.2027, -0.1502, -1.0470])

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_state_value_comparison(net, true_values, x_range, y_range, resolution=50):
    """
    绘制真实状态值和网络预测值的3D对比图

    参数:
    net: 训练好的神经网络
    true_values: 已知的离散状态值，格式为 {(x, y): value} 或 true_values[x][y]
    x_range: tuple, (x_min, x_max)
    y_range: tuple, (y_min, y_max)
    resolution: 连续曲面的采样分辨率
    """

    # 创建用于连续预测的密集网格
    x_cont = np.linspace(x_range[0], x_range[1], resolution)
    y_cont = np.linspace(y_range[0], y_range[1], resolution)
    X_cont, Y_cont = np.meshgrid(x_cont, y_cont)

    # 计算连续预测值
    Z_pred = []
    with torch.no_grad():
        # 向量化计算以提高效率
        inputs = []

        for i in range(resolution):
            outputs = []
            for j in range(resolution):
                coord = (x_cont[j], y_cont[i])
                inp = State(coord)
                out = net(feature_extractor(inp)).item()
                inputs.append(coord)
                outputs.append(out)
            Z_pred.append(outputs)

        outputs = np.array(outputs)
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        # 重塑为网格
        Z_pred = np.array(Z_pred)

    # 准备真实值（离散整数格点）
    x_int = np.arange(x_range[0], x_range[1] + 1)
    y_int = np.arange(y_range[0], y_range[1] + 1)
    X_int, Y_int = np.meshgrid(x_int, y_int)

    Z_true = np.zeros_like(X_int, dtype=float)
    true_x, true_y, true_z = [], [], []

    # 提取真实值
    for i, x in enumerate(x_int):
        for j, y in enumerate(y_int):
            # 根据true_values的数据结构提取值
            if isinstance(true_values, dict):
                # 格式为 {(x, y): value}
                z_val = true_values.get((x, y), 0)
            elif isinstance(true_values[0], (list, np.ndarray, dict)):
                # 格式为 true_values[x][y]
                if isinstance(true_values[x], dict):
                    z_val = true_values[x].get(y, 0)
                else:
                    z_val = true_values[x][y]
            else:
                # 其他格式，可能需要根据你的数据结构调整
                z_val = 0

            Z_true[j, i] = z_val
            true_x.append(x)
            true_y.append(y)
            true_z.append(z_val)

    # 创建图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制连续预测曲面（半透明，显示下面的点）
    surf = ax.plot_surface(X_cont, Y_cont, Z_pred,
                           cmap='viridis',
                           alpha=0.6,  # 半透明以显示下面的点
                           edgecolor='none',
                           label='Network Prediction (Continuous)')

    # 绘制真实值的离散点（突出显示）
    scatter = ax.scatter(true_x, true_y, true_z,
                         color='red',
                         s=80,  # 点的大小
                         marker='o',  # 圆形标记
                         edgecolors='black',
                         linewidth=1.5,
                         depthshade=True,
                         label='True Values (Discrete Grid Points)')

    # 可选：在真实值点之间连线（显示网格结构）
    # 在x方向连线
    for j in range(len(y_int)):
        x_line = x_int
        y_line = np.full_like(x_int, y_int[j])
        z_line = [true_values.get((x, y_int[j]), 0) if isinstance(true_values, dict)
                  else true_values[x][y_int[j]] for x in x_int]
        ax.plot(x_line, y_line, z_line, 'r-', alpha=0.3, linewidth=1)

    # 在y方向连线
    for i in range(len(x_int)):
        x_line = np.full_like(y_int, x_int[i])
        y_line = y_int
        z_line = [true_values.get((x_int[i], y), 0) if isinstance(true_values, dict)
                  else true_values[x_int[i]][y] for y in y_int]
        ax.plot(x_line, y_line, z_line, 'r-', alpha=0.3, linewidth=1)

    # 设置图形属性
    ax.set_xlabel('X State', fontsize=12)
    ax.set_ylabel('Y State', fontsize=12)
    ax.set_zlabel('State Value', fontsize=12)
    ax.set_title('State Value Function: Network Prediction vs True Values', fontsize=14, pad=20)

    # 添加图例
    ax.legend(loc='upper left')

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Predicted Value')

    # 设置视角以便更好地观察
    ax.view_init(elev=25, azim=-45)

    plt.tight_layout()
    plt.show()

    # 返回数据供进一步分析
    return {
        'continuous_grid': (X_cont, Y_cont, Z_pred),
        'discrete_points': (np.array(true_x), np.array(true_y), np.array(true_z)),
        'prediction_range': (Z_pred.min(), Z_pred.max()),
        'true_range': (np.array(true_z).min(), np.array(true_z).max())
    }


# 如果你的真实值是以numpy数组形式存储的辅助函数
def plot_comparison_from_arrays(net, true_array, resolution=50):
    """
    从numpy数组创建对比图

    参数:
    net: 训练好的神经网络
    true_array: 2D numpy数组，true_array[x][y] = value
    resolution: 连续曲面的采样分辨率
    """

    # 获取数组形状
    x_size, y_size = true_array.shape
    x_range = (0, x_size - 1)
    y_range = (0, y_size - 1)

    # 将数组转换为字典格式
    true_dict = {}
    for x in range(x_size):
        for y in range(y_size):
            true_dict[(x, y)] = true_array[x, y]

    return plot_state_value_comparison(net, true_dict, x_range, y_range, resolution)


# 使用示例
if __name__ == "__main__":
    # v(State(1)) = 7.289740085601807
    # v(State(2)) = 8.099737167358398
    # v(State(3)) = 7.999742031097412
    # v(State(4)) = 8.099737167358398
    # v(State(5)) = 8.999730110168457
    # v(State(6)) = 9.99974250793457
    # v(State(7)) = 8.999730110168457
    # v(State(8)) = 9.99974250793457
    # v(State(9)) = 9.99974250793457
    true_values = {
        (0, 0): 7.289740085601807,
        (1, 0): 8.099737167358398,
        (2, 0): 7.999742031097412,
        (0, 1): 8.099737167358398,
        (1, 1): 8.999730110168457,
        (2, 1): 9.99974250793457,
        (0, 2): 8.999730110168457,
        (1, 2): 9.99974250793457,
        (2, 2): 9.99974250793457
    }

    # 绘制对比图
    results = plot_state_value_comparison(net, true_values, (0, 2), (0, 2), resolution=30)

    # 打印统计信息
    print(f"Prediction range: {results['prediction_range']}")
    print(f"True values range: {results['true_range']}")
