#discrete physics-informed neural networks （DPINN）for geometrically nonlinear topology optimization
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

# ==========================================
# 1. 超参数与物理参数设置
# ==========================================
nelx, nely = 180, 60      # 网格划分 (相当于悬臂梁)
lx, ly = 180, 60          # 物理尺寸，长宽比 3:1
dx, dy = lx/nelx, ly/nely # 单元尺寸
volfrac = 0.5             # 目标体积约束
penalty = 3.0             # SIMP惩罚因子
rmin = 2.0                # 密度滤波半径

E0 = 1.0                  # 实体杨氏模量
Emin = 1e-4               # 空洞杨氏模量 (防止刚度矩阵奇异)
nu = 0.3                  # 泊松比
load_val = -0.05          # 载荷 (施加于右侧中间)

# dPINN 训练参数
epochs_init = 400         # 第1步优化的网络训练次数
epochs_step = 50          # 后续每步优化的网络训练次数(热启动)
lr = 2e-3                 # 学习率
max_topopt_iter = 200     # 拓扑优化总迭代次数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 2. 有限元网格与 4 点高斯积分准备
# ==========================================
num_nodes_x, num_nodes_y = nelx + 1, nely + 1
num_nodes = num_nodes_x * num_nodes_y

# 节点坐标 (以左下角为原点)
x_coords = np.linspace(0, lx, num_nodes_x)
y_coords = np.linspace(0, ly, num_nodes_y)
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
nodes = np.vstack([X.flatten(), Y.flatten()]).T  # (N_nodes, 2)
nodes_tensor = torch.tensor(nodes, dtype=torch.float32, device=device)

# 单元连接矩阵 (4节点四边形)
edofMat = np.zeros((nelx * nely, 4), dtype=int)
for i in range(nelx):
    for j in range(nely):
        el = i * nely + j
        n1 = i * num_nodes_y + j
        n2 = (i + 1) * num_nodes_y + j
        edofMat[el] = [n1, n2, n2 + 1, n1 + 1]
edof_tensor = torch.tensor(edofMat, dtype=torch.long, device=device)

# 平面应力弹性矩阵 D0
coef = E0 / (1 - nu**2)
D0 = coef * torch.tensor([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]], dtype=torch.float32, device=device)

# 4点高斯积分的形函数偏导数
gp = 1.0 / np.sqrt(3.0)
gauss_points = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
dN_dx_list = []
detJ_weight_list = []

for xi, eta in gauss_points:
    dN_dxi = np.array([[-(1-eta),  (1-eta), (1+eta), -(1+eta)],
                       [-(1-xi), -(1+xi),  (1+xi),  (1-xi)]]) / 4.0
    J = np.array([[dx/2, 0], [0, dy/2]])
    invJ = np.linalg.inv(J)
    dN_dx = invJ @ dN_dxi  # (2, 4)
    detJ = np.linalg.det(J)
    
    dN_dx_list.append(torch.tensor(dN_dx, dtype=torch.float32, device=device))
    detJ_weight_list.append(detJ * 1.0 * 1.0)

# 预计算密度滤波器 H 矩阵
print("正在构建滤波器矩阵...")
H = torch.zeros((nelx*nely, nelx*nely), device=device)
Hs = torch.zeros(nelx*nely, device=device)
for i in range(nelx):
    for j in range(nely):
        e1 = i * nely + j
        for k in range(max(0, int(i - rmin)), min(nelx, int(i + rmin + 1))):
            for l in range(max(0, int(j - rmin)), min(nely, int(j + rmin + 1))):
                e2 = k * nely + l
                dist = np.sqrt((i - k)**2 + (j - l)**2)
                weight = max(0.0, rmin - dist)
                H[e1, e2] = weight
                Hs[e1] += weight

# ==========================================
# 3. dPINN 神经网络架构
# ==========================================
class dPINN(nn.Module):
    def __init__(self):
        super(dPINN, self).__init__()
        self.B = nn.Parameter(torch.randn(2, 64) * 1.5, requires_grad=False)
        self.net = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64),  nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x_phys):
        x_norm = (x_phys[:, 0:1] - lx/2) / (lx/2)
        y_norm = (x_phys[:, 1:2] - ly/2) / (ly/2)
        x_in = torch.cat([x_norm, y_norm], dim=1)
        
        x_proj = 2.0 * np.pi * torch.matmul(x_in, self.B)
        x_feat = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        u_pred = self.net(x_feat)
        
        # 左侧固定边界条件 (x=0处 u=0)
        mask = (x_phys[:, 0:1] / lx)
        u_constrained = u_pred * mask
        return u_constrained

# ==========================================
# 4. 能量计算与评估 (新增：Von Mises应力计算)
# ==========================================
def evaluate_physics(model, rho):
    u_nodes = model(nodes_tensor)
    u_elems = u_nodes[edof_tensor]

    total_strain_energy = 0.0
    base_element_energy = torch.zeros(nelx*nely, device=device)
    element_von_mises = torch.zeros(nelx*nely, device=device) # 新增：记录单元应力

    I = torch.eye(2, device=device).unsqueeze(0).expand(nelx*nely, -1, -1)

    for gp_idx in range(4):
        dN_dx_g = dN_dx_list[gp_idx]
        weight_g = detJ_weight_list[gp_idx]
        
        grad_u = torch.matmul(u_elems.transpose(1, 2), dN_dx_g.T) 
        F = I + grad_u
        
        Ft = F.transpose(1, 2)
        # Green-Lagrange 应变
        E_tensor = 0.5 * (torch.bmm(Ft, F) - I)
        
        E11 = E_tensor[:, 0, 0]
        E22 = E_tensor[:, 1, 1]
        E12 = E_tensor[:, 0, 1]
        E_voigt = torch.stack([E11, E22, 2*E12], dim=1)
        
        # PK2 应力
        S_voigt = torch.matmul(E_voigt, D0.T)
        W_gp = 0.5 * torch.sum(E_voigt * S_voigt, dim=1)
        
        # 累加高斯点应变能
        base_element_energy += W_gp * weight_g
        
        # 计算 Von Mises 应力 (2D平面应力)
        S11, S22, S12 = S_voigt[:, 0], S_voigt[:, 1], S_voigt[:, 2]
        vm_stress = torch.sqrt(S11**2 + S22**2 - S11*S22 + 3*S12**2)
        element_von_mises += vm_stress / 4.0 # 取4个高斯点的平均值

    # SIMP 惩罚插值
    E_mat = Emin + (rho.flatten() ** penalty) * (1.0 - Emin)
    element_strain_energy = base_element_energy * E_mat
    total_strain_energy = torch.sum(element_strain_energy)
    
    # 真实等效应力 (材料稀疏处应力趋近于0)
    element_von_mises = element_von_mises * (rho.flatten() ** penalty)

    mid_node_y = nely // 2
    load_node_idx = (nelx) * num_nodes_y + mid_node_y
    external_work = u_nodes[load_node_idx, 1] * load_val

    Pi = total_strain_energy - external_work
    return Pi, total_strain_energy, base_element_energy, u_nodes, element_von_mises

# ==========================================
# 5. OC 更新算法
# ==========================================
def oc_update(rho, dc, volfrac):
    l1, l2 = 0.0, 1e9
    move = 0.2
    rho_new = torch.zeros_like(rho)

    while (l2 - l1) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        B_k = -dc / lmid
        rho_new = torch.max(torch.tensor(0.001, device=device), 
                  torch.max(rho - move, 
                  torch.min(torch.tensor(1.0, device=device), 
                  torch.min(rho + move, rho * torch.sqrt(torch.abs(B_k))))))
        if torch.mean(rho_new) - volfrac > 0:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new

# ==========================================
# 6. 主循环与动态可视化 (扩充版 2x2 面板)
# ==========================================
def main():
    plt.ion()  # 开启交互模式（用于实时动态绘图）
    # 创建 2x2 子图
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2 = axs[0, 0], axs[0, 1]
    ax3, ax4 = axs[1, 0], axs[1, 1]
    fig.tight_layout(pad=4.0)

    # 布局定位器，确保 colorbar 不挤压主图
    divs = [make_axes_locatable(ax) for ax in [ax1, ax2, ax3, ax4]]
    caxs = [div.append_axes("right", size="5%", pad=0.1) for div in divs]
    caxs[0].axis('off') # 图1不需要colorbar
    caxs[3].axis('off') # 图4用于画曲线，不需要colorbar

    model = dPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rho = torch.ones(nelx * nely, dtype=torch.float32, device=device) * volfrac
    
    # 历史记录
    history_obj = []
    history_vol = []
    history_chg = []

    print("开始 dPINN 几何非线性拓扑优化...")
    start_time = time.time()

    for opt_iter in range(max_topopt_iter):
        epochs = epochs_init if opt_iter == 0 else epochs_step
        model.train()
        for e in range(epochs):
            optimizer.zero_grad()
            Pi, _, _, _, _ = evaluate_physics(model, rho)
            Pi.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            _, strain_energy, base_energy, u_nodes, vm_stress = evaluate_physics(model, rho)
            obj_val = strain_energy.item() * 2.0
            
            # 计算敏感度
            dc = - penalty * (rho ** (penalty - 1)) * (1.0 - Emin) * base_energy
            dc_filt = torch.matmul(H, rho * dc) / (Hs * torch.max(rho, torch.tensor(1e-3, device=device)))
            
        rho_new = oc_update(rho, dc_filt, volfrac)
        
        # 计算参数变化率
        change = torch.max(torch.abs(rho_new - rho)).item()
        vol_curr = rho_new.mean().item()
        rho = rho_new
        
        # 记录数据
        history_obj.append(obj_val)
        history_vol.append(vol_curr)
        history_chg.append(change)
        
        # ================= 动态可视化 =================
        for ax, cax in zip([ax1, ax2, ax3, ax4], caxs):
            ax.clear()
            if cax != caxs[0] and cax != caxs[3]:
                cax.clear()
        
        # 图1：拓扑图 (材料分布)
        rho_plot = rho.view(nelx, nely).detach().cpu().numpy().T
        ax1.imshow(1 - rho_plot, cmap='gray', origin='lower', extent=[0, lx, 0, ly], vmin=0, vmax=1)
        ax1.set_title(f'Iteration: {opt_iter+1} | Density Distribution')
        ax1.set_aspect('equal', adjustable='box')
        
        # 图2：位移云图 (在变形后的网格上)
        u_out = u_nodes.detach().cpu().numpy()
        u_mag = np.sqrt(u_out[:,0]**2 + u_out[:,1]**2)
        x_def = nodes[:,0] + u_out[:,0]
        y_def = nodes[:,1] + u_out[:,1]
        
        levels_u = np.linspace(0, max(u_mag.max(), 1e-5), 50)
        cf_u = ax2.tricontourf(x_def, y_def, u_mag, levels=levels_u, cmap='jet')
        ax2.set_title(f'Displacement Magnitude (Max: {u_mag.max():.3e})')
        ax2.set_aspect('equal', adjustable='box')
        fig.colorbar(cf_u, cax=caxs[1])
        
        # 图3：Von Mises 等效应力图
        vm_plot = vm_stress.view(nelx, nely).detach().cpu().numpy().T
        cf_s = ax3.imshow(vm_plot, cmap='hot', origin='lower', extent=[0, lx, 0, ly])
        ax3.set_title(f'Von Mises Stress (Max: {vm_plot.max():.3e})')
        ax3.set_aspect('equal', adjustable='box')
        fig.colorbar(cf_s, cax=caxs[2])
        
        # 图4：实时收敛曲线监控 (柔顺度与密度变化率)
        ax4_2 = ax4.twinx() if opt_iter == 0 else ax4_2 # 使用双Y轴
        ax4_2.clear()
        
        ax4.plot(history_obj, 'r-', linewidth=2, label='Compliance (Obj)')
        ax4_2.plot(history_chg, 'b--', linewidth=1.5, label='Max Δρ')
        
        ax4.set_title('Convergence History')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Compliance', color='r')
        ax4_2.set_ylabel('Max Δρ', color='b')
        ax4.set_xlim([0, max_topopt_iter])
        ax4.grid(True, linestyle=':', alpha=0.6)
        
        plt.pause(0.01)
        print(f"Iter: {opt_iter+1:03d} | Obj: {obj_val:.4e} | Vol: {vol_curr:.3f} | Δρ: {change:.4f}")

    print(f"优化完成！总耗时: {time.time() - start_time:.2f} 秒")

    # ==========================================
    # 7. 优化结束：输出二值化（Binarized）清晰设计图
    # ==========================================
    plt.figure(figsize=(10, 4))
    rho_final = rho.view(nelx, nely).detach().cpu().numpy().T
    rho_binary = np.where(rho_final > 0.5, 1.0, 0.0)  # 二值化处理
    plt.imshow(1 - rho_binary, cmap='gray', origin='lower', extent=[0, lx, 0, ly])
    plt.title('Final Binarized Design (ρ > 0.5)')
    plt.axis('equal')

    # ====================== 关键修改：永久保持窗口打开 ======================
    plt.ioff()   # 关闭交互模式
    plt.show()   # 阻塞主线程，所有窗口永久不关闭
    # ======================================================================

if __name__ == '__main__':
    main()