# GNN-PINN with Multiple Random Holes Topology Optimization
# GNN 的图结构天然适应非规则边界：孔洞处 nodal_rho=0，消息传递自动减弱

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

# ==========================================
# 1. 超参数与物理参数
# ==========================================
nelx, nely = 180, 60
lx, ly = 180.0, 60.0
dx, dy = lx/nelx, ly/nely
volfrac = 0.5
penalty = 3.0
rmin = 2.0

E0 = 1.0
Emin = 1e-4
nu = 0.3
load_val = -0.05

epochs_init = 200
epochs_step = 20
lr = 2e-3
max_topopt_iter = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ===== 新增：5 个随机分布圆孔 =====
holes = [
    (45, 15, 8),
    (90, 30, 10),
    (135, 45, 7),
    (60, 40, 6),
    (120, 20, 9),
]
# ===================================

# ==========================================
# 2. 有限元网格与高斯积分
# ==========================================
num_nodes_x, num_nodes_y = nelx + 1, nely + 1
num_nodes = num_nodes_x * num_nodes_y

x_coords = np.linspace(0, lx, num_nodes_x)
y_coords = np.linspace(0, ly, num_nodes_y)
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
nodes = np.vstack([X.flatten(), Y.flatten()]).T
nodes_tensor = torch.tensor(nodes, dtype=torch.float32, device=device)

edofMat = np.zeros((nelx * nely, 4), dtype=int)
for i in range(nelx):
    for j in range(nely):
        el = i * nely + j
        n1 = i * num_nodes_y + j
        n2 = (i + 1) * num_nodes_y + j
        edofMat[el] = [n1, n2, n2 + 1, n1 + 1]
edof_tensor = torch.tensor(edofMat, dtype=torch.long, device=device)

D0 = (E0 / (1 - nu**2)) * torch.tensor([[1, nu, 0],[nu, 1, 0],[0, 0, (1-nu)/2]], dtype=torch.float32, device=device)

gp = 1.0 / np.sqrt(3.0)
gauss_points = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
dN_dx_list = []
detJ_weight_list = []
for xi, eta in gauss_points:
    dN_dxi = np.array([[-(1-eta), (1-eta), (1+eta), -(1+eta)],[-(1-xi), -(1+xi), (1+xi), (1-xi)]]) / 4.0
    J = np.array([[dx/2, 0],[0, dy/2]])
    dN_dx = np.linalg.inv(J) @ dN_dxi
    detJ = np.linalg.det(J)
    dN_dx_list.append(torch.tensor(dN_dx, dtype=torch.float32, device=device))
    detJ_weight_list.append(detJ * 1.0 * 1.0)

# ===== 新增：多孔被动空洞掩码 =====
print("正在构建多孔被动约束掩码...")
cx_elems = (torch.arange(nelx, device=device) + 0.5) * dx
cy_elems = (torch.arange(nely, device=device) + 0.5) * dy
CX_elems, CY_elems = torch.meshgrid(cx_elems, cy_elems, indexing='ij')
passive_void = torch.zeros(nelx * nely, dtype=torch.bool, device=device)
for (center_x, center_y, radius) in holes:
    dist_sq = (CX_elems - center_x)**2 + (CY_elems - center_y)**2
    passive_void |= (dist_sq <= radius**2).flatten()
# ===================================

# 密度滤波器
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
# 3. GNN 图结构
# ==========================================
print("正在构建 GNN 稀疏邻接矩阵...")
i_idx, j_idx = [], []
for e in range(nelx * nely):
    for n in edofMat[e]:
        i_idx.append(n)
        j_idx.append(e)
E2N_idx = torch.tensor([i_idx, j_idx], dtype=torch.long, device=device)
E2N_val = torch.ones(len(i_idx), dtype=torch.float32, device=device)
E2N = torch.sparse_coo_tensor(E2N_idx, E2N_val, size=(num_nodes, nelx*nely))
node_degrees = torch.sparse.sum(E2N, dim=1).to_dense().unsqueeze(1)

A_idx = []
for e in range(nelx * nely):
    nodes_e = edofMat[e]
    for n1 in nodes_e:
        for n2 in nodes_e:
            A_idx.append([n1, n2])
A_idx = np.unique(np.array(A_idx).T, axis=1)
A_idx_t = torch.tensor(A_idx, dtype=torch.long, device=device)
A_val = torch.ones(A_idx_t.shape[1], dtype=torch.float32, device=device)
A_sparse = torch.sparse_coo_tensor(A_idx_t, A_val, size=(num_nodes, num_nodes))
D = torch.sparse.sum(A_sparse, dim=1).to_dense()
D_inv_sqrt = 1.0 / torch.sqrt(D)
row, col = A_idx_t[0], A_idx_t[1]
A_hat_val = D_inv_sqrt[row] * D_inv_sqrt[col]
A_hat = torch.sparse_coo_tensor(A_idx_t, A_hat_val, size=(num_nodes, num_nodes)).coalesce()

# ==========================================
# 4. GNN-dPINN 网络
# ==========================================
class GNN_dPINN(nn.Module):
    def __init__(self, A_hat):
        super(GNN_dPINN, self).__init__()
        self.A_hat = A_hat
        self.gcn1 = nn.Linear(3, 32)
        self.gcn2 = nn.Linear(32, 32)
        self.B = nn.Parameter(torch.randn(2, 64) * 1.5, requires_grad=False)
        self.net = nn.Sequential(
            nn.Linear(160, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x_phys, nodal_rho):
        x_norm = (x_phys[:, 0:1] - lx/2) / (lx/2)
        y_norm = (x_phys[:, 1:2] - ly/2) / (ly/2)
        gcn_in = torch.cat([x_norm, y_norm, nodal_rho], dim=1)
        h1 = torch.sparse.mm(self.A_hat, self.gcn1(gcn_in))
        h1 = torch.tanh(h1)
        z_graph = torch.sparse.mm(self.A_hat, self.gcn2(h1))
        z_graph = torch.tanh(z_graph)
        x_in = torch.cat([x_norm, y_norm], dim=1)
        x_proj = 2.0 * np.pi * torch.matmul(x_in, self.B)
        x_rff = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        feat = torch.cat([x_rff, z_graph], dim=-1)
        u_pred = self.net(feat)
        mask = (x_phys[:, 0:1] / lx)
        return u_pred * mask, z_graph

# ==========================================
# 5. 能量计算
# ==========================================
def evaluate_physics(model, rho, nodal_rho):
    u_nodes, z_graph = model(nodes_tensor, nodal_rho)
    u_elems = u_nodes[edof_tensor]

    total_strain_energy = 0.0
    base_element_energy = torch.zeros(nelx*nely, device=device)
    element_von_mises = torch.zeros(nelx*nely, device=device)

    I = torch.eye(2, device=device).unsqueeze(0).expand(nelx*nely, -1, -1)

    for gp_idx in range(4):
        dN_dx_g = dN_dx_list[gp_idx]
        weight_g = detJ_weight_list[gp_idx]
        grad_u = torch.matmul(u_elems.transpose(1, 2), dN_dx_g.T)
        F = I + grad_u
        Ft = F.transpose(1, 2)
        E_tensor = 0.5 * (torch.bmm(Ft, F) - I)
        E11, E22, E12 = E_tensor[:, 0, 0], E_tensor[:, 1, 1], E_tensor[:, 0, 1]
        E_voigt = torch.stack([E11, E22, 2*E12], dim=1)
        S_voigt = torch.matmul(E_voigt, D0.T)
        base_element_energy += 0.5 * torch.sum(E_voigt * S_voigt, dim=1) * weight_g
        S11, S22, S12 = S_voigt[:, 0], S_voigt[:, 1], S_voigt[:, 2]
        vm = torch.sqrt(S11**2 + S22**2 - S11*S22 + 3*S12**2)
        element_von_mises += vm / 4.0

    E_mat = Emin + (rho.flatten() ** penalty) * (1.0 - Emin)
    element_strain_energy = base_element_energy * E_mat
    total_strain_energy = torch.sum(element_strain_energy)
    element_von_mises = element_von_mises * (rho.flatten() ** penalty)

    mid_node_y = nely // 2
    load_node_idx = (nelx) * num_nodes_y + mid_node_y
    external_work = u_nodes[load_node_idx, 1] * load_val

    Pi = total_strain_energy - external_work
    return Pi, total_strain_energy, base_element_energy, u_nodes, element_von_mises, z_graph

# ==========================================
# 6. OC 更新
# ==========================================
def oc_update(rho, dc, volfrac, passive_void):
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
        rho_new[passive_void] = 0.001
        if torch.mean(rho_new) - volfrac > 0:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new

# ==========================================
# 7. 主循环
# ==========================================
def main():
    plt.ion()
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    ax1, ax2, ax3 = axs[0, 0], axs[0, 1], axs[0, 2]
    ax4, ax5, ax6 = axs[1, 0], axs[1, 1], axs[1, 2]
    fig.tight_layout(pad=4.0)
    divs = [make_axes_locatable(ax) for ax in [ax1, ax2, ax3, ax4, ax5, ax6]]
    caxs = [div.append_axes("right", size="5%", pad=0.1) for div in divs]
    caxs[0].axis('off')
    caxs[3].axis('off')

    model = GNN_dPINN(A_hat).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rho = torch.ones(nelx * nely, dtype=torch.float32, device=device) * volfrac
    rho[passive_void] = 0.001

    history_obj, history_vol, history_chg = [], [], []

    print("开始 GNN-PINN + 多孔几何 拓扑优化...")
    print(f"[优势] 共 {len(holes)} 个圆孔，GNN 的图结构天然适应非规则边界")
    start_time = time.time()

    for opt_iter in range(max_topopt_iter):
        nodal_rho = torch.sparse.mm(E2N, rho.unsqueeze(1)) / node_degrees

        epochs = epochs_init if opt_iter == 0 else epochs_step
        model.train()
        for e in range(epochs):
            optimizer.zero_grad()
            Pi, _, _, _, _, _ = evaluate_physics(model, rho, nodal_rho)
            Pi.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            _, strain_energy, base_energy, u_nodes, vm_stress, z_graph = evaluate_physics(model, rho, nodal_rho)
            obj_val = strain_energy.item() * 2.0
            dc = - penalty * (rho ** (penalty - 1)) * (1.0 - Emin) * base_energy
            dc_filt = torch.matmul(H, rho * dc) / (Hs * torch.max(rho, torch.tensor(1e-3, device=device)))

        rho_new = oc_update(rho, dc_filt, volfrac, passive_void)
        change = torch.max(torch.abs(rho_new - rho)).item()
        vol_curr = rho_new.mean().item()
        rho = rho_new

        history_obj.append(obj_val)
        history_vol.append(vol_curr)
        history_chg.append(change)

        for ax, cax in zip([ax1, ax2, ax3, ax4, ax5, ax6], caxs):
            ax.clear()
            if cax not in [caxs[0], caxs[3]]:
                cax.clear()

        rho_plot = rho.view(nelx, nely).detach().cpu().numpy().T
        ax1.imshow(1 - rho_plot, cmap='gray', origin='lower', extent=[0, lx, 0, ly], vmin=0, vmax=1)
        ax1.set_title(f'GNN-MultiHole | Iter: {opt_iter+1}')
        ax1.set_aspect('equal')

        u_out = u_nodes.detach().cpu().numpy()
        u_mag = np.sqrt(u_out[:,0]**2 + u_out[:,1]**2)
        x_def, y_def = nodes[:,0] + u_out[:,0], nodes[:,1] + u_out[:,1]
        levels_u = np.linspace(0, max(u_mag.max(), 1e-5), 50)
        cf_u = ax2.tricontourf(x_def, y_def, u_mag, levels=levels_u, cmap='jet')
        ax2.set_title(f'Displacement (Max: {u_mag.max():.3e})')
        ax2.set_aspect('equal')
        fig.colorbar(cf_u, cax=caxs[1])

        vm_plot = vm_stress.view(nelx, nely).detach().cpu().numpy().T
        cf_s = ax3.imshow(vm_plot, cmap='hot', origin='lower', extent=[0, lx, 0, ly])
        ax3.set_title(f'Von Mises Stress')
        ax3.set_aspect('equal')
        fig.colorbar(cf_s, cax=caxs[2])

        ax4_2 = ax4.twinx() if opt_iter == 0 else ax4_2
        ax4_2.clear()
        ax4.plot(history_obj, 'r-', linewidth=2, label='Compliance')
        ax4_2.plot(history_chg, 'b--', linewidth=1.5, label='Max Δρ')
        ax4.set_title('Convergence History')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Compliance', color='r')
        ax4_2.set_ylabel('Max Δρ', color='b')
        ax4.set_xlim([0, max_topopt_iter])
        ax4.grid(True, linestyle=':', alpha=0.6)

        feat_ch0 = z_graph[:, 0].view(num_nodes_x, num_nodes_y).detach().cpu().numpy().T
        cf_f0 = ax5.imshow(feat_ch0, cmap='viridis', origin='lower', extent=[0, lx, 0, ly])
        ax5.set_title('GNN Latent Feature (Ch 0)')
        ax5.set_aspect('equal')
        fig.colorbar(cf_f0, cax=caxs[4])

        feat_ch1 = z_graph[:, 1].view(num_nodes_x, num_nodes_y).detach().cpu().numpy().T
        cf_f1 = ax6.imshow(feat_ch1, cmap='magma', origin='lower', extent=[0, lx, 0, ly])
        ax6.set_title('GNN Latent Feature (Ch 1)')
        ax6.set_aspect('equal')
        fig.colorbar(cf_f1, cax=caxs[5])

        plt.pause(0.01)
        print(f"Iter: {opt_iter+1:03d} | Obj: {obj_val:.4e} | Vol: {vol_curr:.3f} | Δρ: {change:.4f}")

    print(f"优化完成！总耗时: {time.time() - start_time:.2f} 秒")

    plt.figure(figsize=(10, 4))
    rho_final = rho.view(nelx, nely).detach().cpu().numpy().T
    rho_binary = np.where(rho_final > 0.5, 1.0, 0.0)
    plt.imshow(1 - rho_binary, cmap='gray', origin='lower', extent=[0, lx, 0, ly])
    plt.title('GNN-PINN-MultiHole Final Design')
    plt.axis('equal')

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
