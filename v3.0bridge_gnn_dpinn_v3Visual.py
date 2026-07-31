"""
GNN-dPINN for Bridge Topology Optimization (Truss Bridge)
Scheme A: Quadratic Parabola Arch
Half-structure with updated geometry
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib

# =============================================================================
#  LIVE DISPLAY SWITCH  --  set to False to disable real-time window
# =============================================================================
interactive_mode = True   # <<-- 手动改为 False 即可关闭实时弹窗，仅后台保存 GIF

if not interactive_mode:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import os
import sys
from io import BytesIO
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Fix Windows terminal Chinese encoding & prevent window flash for GIF frames
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

# Setup logging to file
class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', buffering=1, encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

os.makedirs("bridge_results", exist_ok=True)
sys.stdout = Logger("bridge_results/optimization_log.txt")

# ==========================================
# 0. Animation Helpers  (backend-agnostic, no window flash)
# ==========================================
def fig_to_pil_image(fig):
    """Convert a matplotlib Figure to a PIL Image (RGB) — backend agnostic."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi if hasattr(fig, 'dpi') else 100)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img_copy = img.copy()
    buf.close()
    return img_copy


def save_gif(frames, filepath, duration_ms=500, loop=0):
    """Save a list of PIL Images as an animated GIF."""
    if not frames:
        print(f"Warning: no frames to save for {filepath}")
        return
    frames[0].save(
        filepath,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop
    )
    print(f"Saved animation: {filepath} ({len(frames)} frames)")


# ==========================================
# 1. Hyperparameters and Physical Parameters
# ==========================================
nelx, nely = 370, 126       # Mesh resolution for half-structure (1m x 1m elements)
lx, ly = 370.0, 126.0       # Physical dimensions (half-structure)
dx, dy = lx/nelx, ly/nely   # Element sizes (both should be 1.0)

volfrac = 0.50              # Target volume fraction in design domain (v2)
penalty = 3.0               # Moderate penalty
rmin = 3.0                  # Filter radius

E0 = 1.0                    # Solid Young's modulus
Emin = 1e-4                 # Void Young's modulus
nu = 0.3                    # Poisson's ratio

# Load parameters
total_load = -0.5           # Reduced load for stability

# dPINN training parameters
epochs_init = 500           # Training for initial step
epochs_step = 300           # Training for regular steps
lr = 5e-4                   # Learning rate
max_topopt_iter = 150       # More iterations for sharper topology

# Animation parameters
frame_interval = 1          # Save a frame / update live window every N iterations
gif_duration_ms = 500       # Duration per frame in milliseconds
live_pause_sec = 0.01       # Small pause to let the GUI event loop process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print(f"Device: {device}")
print(f"Half-structure mesh: {nelx} x {nely} = {nelx*nely} elements")
print(f"Physical size: {lx} x {ly}")
print(f"Element size: dx={dx:.4f}, dy={dy:.4f}")
print(f"Interactive live display: {'ON' if interactive_mode else 'OFF'}")

# ==========================================
# 2. Define Fixed Regions and Design Domains
# ==========================================
x_elem = (np.arange(nelx) + 0.5) * dx  # element center x-coordinates
y_elem = (np.arange(nely) + 0.5) * dy  # element center y-coordinates

# Bridge pier parameters
pier_center = 195.0
pier_half_width = 6.0                   # 12m width -> half = 6m
pier_left = pier_center - pier_half_width   # 189.0
pier_right = pier_center + pier_half_width  # 201.0
deck_bottom = 100.0                     # Bridge deck bottom y-coordinate
deck_top = 106.0                        # Bridge deck top y-coordinate

# -------------------------------------------------
# Design Domain I (Lower): lower boundary function
# -------------------------------------------------
def lower_boundary_y(x):
    """
    Lower boundary of Design Domain I.
    Piecewise: diagonal -> horizontal -> parabola (left side)
               parabola -> horizontal (right side)
    v3: horizontal segment raised from y=70 to y=80
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    
    # Left side: diagonal x∈[0,30], from (0,100) to (30,80)
    mask1 = (x >= 0) & (x <= 30)
    result[mask1] = 100.0 - (2.0/3.0) * x[mask1]  # slope = -2/3
    
    # Left side: horizontal x∈[30,95], y=80
    mask2 = (x > 30) & (x <= 95)
    result[mask2] = 80.0
    
    # Left side: parabola x∈[95,189], through (95,80) and (189,40)
    # tangent horizontal at x=95
    mask3 = (x > 95) & (x <= pier_left)
    a_left = -40.0 / (94.0**2)  # 40-80 = -40, span = 94
    result[mask3] = a_left * (x[mask3] - 95.0)**2 + 80.0
    
    # Pier region: not part of design domain I (will be handled by pier_mask)
    mask_pier = (x > pier_left) & (x < pier_right)
    result[mask_pier] = np.nan
    
    # Right side: parabola x∈[201,295], through (201,40) and (295,80)
    # tangent horizontal at x=295
    mask4 = (x >= pier_right) & (x <= 295)
    a_right = -40.0 / (94.0**2)  # 40-80 = -40, span = 94
    result[mask4] = a_right * (x[mask4] - 295.0)**2 + 80.0
    
    # Right side: horizontal x∈[295,370], y=80
    mask5 = (x > 295) & (x <= lx)
    result[mask5] = 80.0
    
    return result

# -------------------------------------------------
# Design Domain II (Upper Arch): upper boundary
# -------------------------------------------------
def arch_upper_boundary_y(x):
    """Upper parabolic arch boundary: y = -0.002*(x-195)^2 + 126"""
    x = np.asarray(x)
    return -0.002 * (x - 195.0)**2 + 126.0

def arch_lower_boundary_y(x):
    """Lower boundary of arch design domain is deck top: y = 106"""
    x = np.asarray(x)
    return np.full_like(x, 106.0)

# -------------------------------------------------
# Build element masks
# -------------------------------------------------

# 2.1 Bridge deck (non-design solid): y ∈ [100, 106]
deck_mask = (y_elem[None, :] >= deck_bottom) & (y_elem[None, :] <= deck_top)
deck_mask = np.tile(deck_mask, (nelx, 1))  # shape (nelx, nely)

# 2.2 Bridge pier (partially non-design solid):
# - y∈[0,40]: fixed solid (base)
# - y∈[40,100]: design domain (released for topology optimization)
pier_fixed_mask = (np.abs(x_elem[:, None] - pier_center) < pier_half_width) & \
                  (y_elem[None, :] <= 40.0)
pier_design_mask = (np.abs(x_elem[:, None] - pier_center) < pier_half_width) & \
                   (y_elem[None, :] > 40.0) & (y_elem[None, :] <= 100.0)
# Full pier mask for exclusion in design domain I
pier_mask = pier_fixed_mask | pier_design_mask

# 2.3 Design Domain I (lower): elements between lower_boundary_y and deck_bottom
# Exclude pier region
boundary_y_at_x = lower_boundary_y(x_elem)  # shape (nelx,)
lower_boundary_grid = boundary_y_at_x[:, None]  # shape (nelx, 1)

# Design Domain I: y between lower boundary and deck bottom (100)
# AND not in pier region
y_elem_grid = y_elem[None, :]  # shape (1, nely)
design1_mask = y_elem_grid >= lower_boundary_grid
# But we need y < deck_bottom because deck is separate
design1_mask &= (y_elem_grid < deck_bottom)
design1_mask &= ~pier_mask  # exclude pier
# Also need valid boundary (not nan)
valid_lower = ~np.isnan(boundary_y_at_x)[:, None]
design1_mask &= valid_lower

# 2.4 Design Domain II (upper arch): x∈[95,295], y between 106 and arch top
arch_top_at_x = arch_upper_boundary_y(x_elem)  # shape (nelx,)
arch_top_grid = arch_top_at_x[:, None]
arch_bottom_grid = np.full_like(arch_top_grid, deck_top)

# Arch design domain: x in [95, 295], y between 106 and arch top
x_in_arch = (x_elem[:, None] >= 95.0) & (x_elem[:, None] <= 295.0)
y_in_arch = (y_elem[None, :] >= arch_bottom_grid) & (y_elem[None, :] <= arch_top_grid)
design2_mask = x_in_arch & y_in_arch

# 2.5 Fixed void regions:
# (a) Below lower boundary (Design Domain I void)
void_below_design1 = (y_elem_grid < lower_boundary_grid) & valid_lower & ~pier_mask

# (b) Above deck (y>106) but outside arch domain
# Left side: x<95, y∈[106,126]
void_arch_left = (x_elem[:, None] < 95.0) & (y_elem_grid > deck_top)
# Right side: x>295, y∈[106,126]
void_arch_right = (x_elem[:, None] > 295.0) & (y_elem_grid > deck_top)
# Above arch top: x∈[95,295], y > arch_top
void_arch_above = (x_elem[:, None] >= 95.0) & (x_elem[:, None] <= 295.0) & \
                  (y_elem_grid > arch_top_grid)

void_mask = void_below_design1 | void_arch_left | void_arch_right | void_arch_above

# 2.6 Combine fixed regions (non-design solid)
fixed_mask = np.zeros((nelx, nely), dtype=bool)
fixed_mask |= deck_mask
fixed_mask |= pier_fixed_mask  # Only pier base (y<=40) is fixed solid

fixed_mask_flat = fixed_mask.flatten()
void_mask_flat = void_mask.flatten()
design1_mask_flat = design1_mask.flatten()
design2_mask_flat = design2_mask.flatten()

fixed_elements = np.where(fixed_mask_flat)[0]
void_elements = np.where(void_mask_flat)[0]
design1_elements = np.where(design1_mask_flat)[0]
design2_elements = np.where(design2_mask_flat)[0]
# Total design elements = design1 + design2
design_elements = np.where(~fixed_mask_flat & ~void_mask_flat)[0]

n_fixed = len(fixed_elements)
n_void = len(void_elements)
n_design1 = len(design1_elements)
n_design2 = len(design2_elements)
n_design = len(design_elements)
n_total = nelx * nely

print(f"Fixed elements: {n_fixed} ({n_fixed/n_total*100:.1f}%)")
print(f"  - Deck: {np.sum(deck_mask)} elements")
print(f"  - Pier: {np.sum(pier_mask)} elements")
print(f"Void elements: {n_void} ({n_void/n_total*100:.1f}%)")
print(f"Design Domain I (lower): {n_design1} ({n_design1/n_total*100:.1f}%)")
print(f"Design Domain II (arch): {n_design2} ({n_design2/n_total*100:.1f}%)")
print(f"Total Design elements: {n_design} ({n_design/n_total*100:.1f}%)")

# Volume constraint: target fraction applies to design domains only
# Fixed elements are always solid (density=1.0)
# Design elements have target volume fraction = volfrac
target_volume = n_fixed + volfrac * n_design
design_initial_density = volfrac

print(f"Fillable elements: {n_fixed + n_design} ({(n_fixed + n_design)/n_total*100:.1f}%)")
print(f"Target total volume: {target_volume:.0f} (overall frac: {target_volume/n_total:.4f})")
print(f"Design domain target volume: {volfrac * n_design:.0f}")
print(f"Design domain initial density: {design_initial_density:.4f}")

# ==========================================
# 3. Node Coordinates and Boundary Conditions
# ==========================================
num_nodes_x, num_nodes_y = nelx + 1, nely + 1
num_nodes = num_nodes_x * num_nodes_y

x_coords = np.linspace(0, lx, num_nodes_x)
y_coords = np.linspace(0, ly, num_nodes_y)
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
nodes = np.vstack([X.flatten(), Y.flatten()]).T
nodes_tensor = torch.tensor(nodes, dtype=torch.float32, device=device)

# Boundary condition masks
# Left roller support: x=0, y=100 (deck bottom)
left_pin_nodes = (nodes[:, 0] < 1e-6) & (np.abs(nodes[:, 1] - 100.0) < 1e-6)

# Symmetry boundary: x=370, all y
right_roller_nodes = (nodes[:, 0] > lx - 1e-6)

# Pier base fixed: y=0, x∈[189,201] (only base, y<=40)
pier_base_nodes = (np.abs(nodes[:, 0] - pier_center) < pier_half_width + 1e-6) & (nodes[:, 1] <= 40.0)

bc_type = np.zeros(num_nodes, dtype=np.int32)
bc_type[left_pin_nodes | pier_base_nodes] = 3  # fixed both directions
bc_type[right_roller_nodes] = 2                # fixed x direction (symmetry)

print(f"Left pin nodes: {np.sum(left_pin_nodes)}")
print(f"Symmetry boundary nodes: {np.sum(right_roller_nodes)}")
print(f"Pier base fixed nodes: {np.sum(pier_base_nodes)}")

# Load nodes: deck top surface y=106
load_node_mask = (np.abs(nodes[:, 1] - deck_top) < 1e-6)
load_node_indices = np.where(load_node_mask)[0]
load_per_node = total_load / len(load_node_indices)
print(f"Load nodes: {len(load_node_indices)}, load per node: {load_per_node:.4f}")

# ==========================================
# 4. Finite Element Mesh and Gauss Quadrature
# ==========================================
edofMat = np.zeros((nelx * nely, 4), dtype=int)
for i in range(nelx):
    for j in range(nely):
        el = i * nely + j
        n1 = i * num_nodes_y + j
        n2 = (i + 1) * num_nodes_y + j
        n3 = (i + 1) * num_nodes_y + (j + 1)
        n4 = i * num_nodes_y + (j + 1)
        edofMat[el] = [n1, n2, n3, n4]
edof_tensor = torch.tensor(edofMat, dtype=torch.long, device=device)

D0 = (E0 / (1 - nu**2)) * torch.tensor([[1, nu, 0],
                                        [nu, 1, 0],
                                        [0, 0, (1-nu)/2]], dtype=torch.float32, device=device)

gp = 1.0 / np.sqrt(3.0)
gauss_points = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
dN_dx_list = []
detJ_weight_list = []

for xi, eta in gauss_points:
    dN_dxi = np.array([[-(1-eta),  (1-eta), (1+eta), -(1+eta)],
                       [-(1-xi), -(1+xi),  (1+xi),  (1-xi)]]) / 4.0
    J = np.array([[dx/2, 0], [0, dy/2]])
    dN_dx = np.linalg.inv(J) @ dN_dxi
    detJ = np.linalg.det(J)
    dN_dx_list.append(torch.tensor(dN_dx, dtype=torch.float32, device=device))
    detJ_weight_list.append(detJ)

# ==========================================
# 5. Density Filter Matrix
# ==========================================
print("Building density filter matrix...")
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
# 6. GNN Sparse Adjacency Matrix
# ==========================================
print("Building GNN sparse adjacency matrix...")
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

bc_uxuy = torch.tensor(bc_type == 3, dtype=torch.bool, device=device)
bc_uy = torch.tensor(bc_type == 2, dtype=torch.bool, device=device)

# ==========================================
# 7. GNN-dPINN Network
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
            nn.Linear(128, 64),  nn.Tanh(),
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
        
        feat_fusion = torch.cat([x_rff, z_graph], dim=-1)
        u_pred = self.net(feat_fusion)
        
        ux = u_pred[:, 0].clone()
        uy = u_pred[:, 1].clone()
        ux = ux * (~bc_uxuy) * (~bc_uy)  # Fixed nodes and symmetry boundary: ux=0
        uy = uy * (~bc_uxuy)              # Only fixed nodes: uy=0
        # Symmetry boundary (bc_uy): ux=0, uy is free
        
        return torch.stack([ux, uy], dim=1), z_graph

# ==========================================
# 8. Physics Evaluation
# ==========================================
def evaluate_physics(model, rho, nodal_rho):
    """Evaluate physics using the design variable rho (original density)."""
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
        # Linear (infinitesimal) strain tensor for consistency with SIMP linear elasticity
        E_tensor = 0.5 * (grad_u + grad_u.transpose(1, 2))
        E11, E22, E12 = E_tensor[:, 0, 0], E_tensor[:, 1, 1], E_tensor[:, 0, 1]
        E_voigt = torch.stack([E11, E22, 2*E12], dim=1)
        S_voigt = torch.matmul(E_voigt, D0.T)
        base_element_energy += 0.5 * torch.sum(E_voigt * S_voigt, dim=1) * weight_g
        S11, S22, S12 = S_voigt[:, 0], S_voigt[:, 1], S_voigt[:, 2]
        vm_stress = torch.sqrt(S11**2 + S22**2 - S11*S22 + 3*S12**2)
        element_von_mises += vm_stress / 4.0

    E_mat = Emin + (rho.flatten() ** penalty) * (1.0 - Emin)
    element_strain_energy = base_element_energy * E_mat
    total_strain_energy = torch.sum(element_strain_energy)
    element_von_mises = element_von_mises * (rho.flatten() ** penalty)

    load_nodes_t = torch.tensor(load_node_indices, dtype=torch.long, device=device)
    external_work = torch.sum(u_nodes[load_nodes_t, 1]) * load_per_node
    Pi = total_strain_energy - external_work
    
    return Pi, total_strain_energy, base_element_energy, u_nodes, element_von_mises, z_graph

# ==========================================
# 9. OC Update
# ==========================================
def oc_update(rho, dc, fixed_elements, void_elements, target_volume):
    l1, l2 = 0.0, 1e9
    move = 0.02               # OC step size
    rho_new = torch.zeros_like(rho)
    iter_count = 0
    while (l2 - l1) > 1e-4 and iter_count < 100:
        lmid = 0.5 * (l1 + l2)
        B_k = -dc / lmid
        rho_new = torch.max(torch.tensor(0.001, device=device),
                  torch.max(rho - move,
                  torch.min(torch.tensor(1.0, device=device),
                  torch.min(rho + move, rho * torch.sqrt(torch.abs(B_k))))))
        rho_new[fixed_elements] = 1.0
        rho_new[void_elements] = 0.0
        current_volume = torch.sum(rho_new).item()
        if current_volume - target_volume > 0:
            l1 = lmid
        else:
            l2 = lmid
        iter_count += 1
    return rho_new

# ==========================================
# 10. Mirror Function
# ==========================================
def mirror_density(rho_half_2d):
    nelx_half, nely_half = rho_half_2d.shape
    rho_full = np.zeros((2 * nelx_half, nely_half))
    rho_full[:nelx_half, :] = rho_half_2d
    rho_full[nelx_half:, :] = rho_half_2d[::-1, :]
    return rho_full

# ==========================================
# 10b. Local Symmetry Constraint
# Enforce symmetry about pier centerline (x=195) ONLY within x∈[95,295]
# Outside this range (x<95 or x>295), topology is free (asymmetric)
# ==========================================
def apply_local_symmetry(rho_2d, x_elem, pier_center=195.0, sym_range=(95.0, 295.0)):
    """
    Apply local symmetry constraint within specified range.
    Only elements within [sym_range[0], sym_range[1]] are symmetrized.
    
    For each element at x in [95, 195], its symmetric partner is at x' = 390 - x in [195, 295].
    We average their densities to enforce rho(x,y) = rho(390-x, y).
    """
    nelx, nely = rho_2d.shape
    rho_sym = rho_2d.copy()
    
    # Find element indices within symmetry range [95, 295]
    sym_mask = (x_elem >= sym_range[0]) & (x_elem <= sym_range[1])
    sym_indices = np.where(sym_mask)[0]
    
    if len(sym_indices) == 0:
        return rho_sym
    
    # For each element in the left part of symmetry range [95, 195],
    # find its mirror in [195, 295] and average densities
    for i in sym_indices:
        x_i = x_elem[i]
        if x_i > pier_center:
            continue  # Only process left half to avoid double processing
        
        x_mirror = 2 * pier_center - x_i  # 390 - x_i
        # Find closest element index for mirror position
        j = np.argmin(np.abs(x_elem - x_mirror))
        
        # Average densities (symmetrize)
        avg = 0.5 * (rho_2d[i, :] + rho_2d[j, :])
        rho_sym[i, :] = avg
        rho_sym[j, :] = avg
    
    return rho_sym

# ==========================================
# 11. Animation Frame Capture Helpers (for GIF export)
# ==========================================
def capture_topology_frame(rho_2d, title, extent, figsize=(12, 4), cmap='gray'):
    """Capture a single topology frame as a PIL Image (no window flash)."""
    fig = Figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.imshow(1 - rho_2d, cmap=cmap, origin='lower', extent=extent, vmin=0, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    return fig_to_pil_image(fig)


def capture_compliance_frame(history_obj, current_iter, figsize=(8, 5)):
    """Capture a convergence curve frame as a PIL Image (no window flash)."""
    fig = Figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    iters = range(1, len(history_obj) + 1)
    ax.plot(iters, history_obj, 'b-', linewidth=2, label='Compliance')
    ax.scatter([current_iter], [history_obj[-1]], color='red', s=60, zorder=5, label='Current')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Compliance', fontsize=11)
    ax.set_title(f'Compliance Convergence - Iteration {current_iter}', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)
    return fig_to_pil_image(fig)


# ==========================================
# 12. Main Optimization
# ==========================================
def main():
    model = GNN_dPINN(A_hat).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    rho = torch.ones(nelx * nely, dtype=torch.float32, device=device) * design_initial_density
    rho[fixed_elements] = 1.0
    rho[void_elements] = 0.0
    
    history_obj, history_vol, history_chg, history_loss = [], [], [], []

    # Animation frame buffers (for GIF export)
    half_topo_frames = []
    full_topo_frames = []
    compliance_frames = []

    # -------------------------------------------------------------------------
    #  REAL-TIME LIVE DISPLAY SETUP
    # -------------------------------------------------------------------------
    fig_live = None
    im_topo = None
    line_conv = None
    point_conv = None
    ax_topo = None
    ax_conv = None

    if interactive_mode:
        plt.ion()                       # turn on interactive mode
        fig_live, (ax_topo, ax_conv) = plt.subplots(1, 2, figsize=(16, 5))
        fig_live.canvas.manager.set_window_title('GNN-dPINN Bridge Topology Optimization')

        # -- left panel: topology --
        rho_init = rho.view(nelx, nely).detach().cpu().numpy().T
        im_topo = ax_topo.imshow(
            1 - rho_init, cmap='gray', origin='lower',
            extent=[0, lx, 0, ly], vmin=0, vmax=1
        )
        ax_topo.set_title('Half-structure Topology', fontsize=12)
        ax_topo.set_xlabel('x')
        ax_topo.set_ylabel('y')
        ax_topo.set_aspect('equal')

        # -- right panel: convergence --
        line_conv, = ax_conv.plot([], [], 'b-', linewidth=2, label='Compliance')
        point_conv = ax_conv.scatter([], [], color='red', s=60, zorder=5)
        ax_conv.set_xlabel('Iteration', fontsize=11)
        ax_conv.set_ylabel('Compliance', fontsize=11)
        ax_conv.set_title('Compliance Convergence', fontsize=12)
        ax_conv.grid(True, linestyle='--', alpha=0.6)
        ax_conv.legend(fontsize=10)

        plt.tight_layout()
        plt.show(block=False)
        fig_live.canvas.draw()
        fig_live.canvas.flush_events()
        print("Live display window opened.")

    print("\n" + "="*60)
    print("Starting GNN-dPINN Bridge Topology Optimization")
    print("Scheme A: Quadratic Parabola Arch")
    print("v3visual: Lower boundary horizontal raised to y=80")
    print("    Pier upper part (y belong [40,100]) design domain, volfrac=0.50")
    print("    Local symmetry within x belong [95,295]")
    print("    Live animation + GIF export enabled")
    print(f"    Frame interval: every {frame_interval} iterations")
    print("="*60)
    start_time = time.time()

    for opt_iter in range(max_topopt_iter):
        # Nodal density for GNN input (based on design variable rho)
        nodal_rho = torch.sparse.mm(E2N, rho.unsqueeze(1)) / node_degrees

        # === Adaptive training epochs ===
        if opt_iter == 0:
            epochs = epochs_init
        elif opt_iter < 20:
            epochs = 300  # More training early when density changes are large
        elif opt_iter < 60:
            epochs = epochs_step
        else:
            epochs = 200  # Later when density is nearly stable
        
        model.train()
        epoch_loss = 0.0
        for e in range(epochs):
            optimizer.zero_grad()
            Pi, strain_energy, base_energy, u_nodes, vm_stress, z_graph = evaluate_physics(model, rho, nodal_rho)
            # For linear elasticity, correct solution satisfies: strain_energy = 0.5 * external_work
            # Add a physics consistency loss to help GNN find physically correct displacement field
            load_nodes_t = torch.tensor(load_node_indices, dtype=torch.long, device=device)
            external_work = torch.sum(u_nodes[load_nodes_t, 1]) * load_per_node
            consistency_loss = (strain_energy - 0.5 * external_work)**2
            loss = Pi + 0.01 * consistency_loss
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()
        
        history_loss.append(epoch_loss)
        
        model.eval()
        with torch.no_grad():
            Pi, strain_energy, base_energy, u_nodes, vm_stress, z_graph = evaluate_physics(model, rho, nodal_rho)
            # Compliance = external work = F^T U (more reliable than 2*strain_energy when KU=F is not exactly satisfied)
            load_nodes_t = torch.tensor(load_node_indices, dtype=torch.long, device=device)
            external_work = torch.sum(u_nodes[load_nodes_t, 1]) * load_per_node
            obj_val = external_work.item()
            
            # Standard sensitivity filtering (using base_energy for SIMP sensitivity)
            dc = -penalty * (rho ** (penalty - 1)) * (1.0 - Emin) * base_energy
            dc_filt = torch.matmul(H, dc) / Hs
        
        rho_new = oc_update(rho, dc_filt, fixed_elements, void_elements, target_volume)
        
        # === Apply LOCAL symmetry constraint about pier centerline (x=195) within [95,295] ===
        # Only symmetrize elements in x∈[95,295]; outside this range, topology remains asymmetric
        rho_new_2d = rho_new.view(nelx, nely).detach().cpu().numpy()
        rho_sym_2d = apply_local_symmetry(rho_new_2d, x_elem, pier_center=195.0, sym_range=(95.0, 295.0))
        rho_new = torch.tensor(rho_sym_2d.flatten(), dtype=torch.float32, device=device)
        # Re-apply fixed and void masks after symmetry operation
        rho_new[fixed_elements] = 1.0
        rho_new[void_elements] = 0.0
        # Re-normalize to maintain volume constraint after symmetry adjustment
        current_vol = torch.sum(rho_new[design_elements]).item()
        target_design_vol = volfrac * n_design
        if current_vol > 0:
            scale = target_design_vol / current_vol
            rho_new[design_elements] = torch.clamp(rho_new[design_elements] * scale, 0.001, 1.0)
        
        change = torch.max(torch.abs(rho_new - rho)).item()
        vol_curr = rho_new.mean().item()
        rho = rho_new
        
        history_obj.append(obj_val)
        history_vol.append(vol_curr)
        history_chg.append(change)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Iter: {opt_iter+1:03d} | Compliance: {obj_val:.4e} | dRho: {change:.4f} | Vol: {vol_curr:.4f} | LR: {current_lr:.2e} | Epochs: {epochs} | Loss: {epoch_loss:.4e}")
        
        scheduler.step()
        
        # -----------------------------------------------------------------
        #  REAL-TIME UPDATE  +  GIF FRAME CAPTURE
        # -----------------------------------------------------------------
        should_capture = (opt_iter + 1) % frame_interval == 0 or opt_iter == max_topopt_iter - 1
        
        if should_capture:
            rho_half_2d = rho.view(nelx, nely).detach().cpu().numpy().T
            
            # -- live window update --
            if interactive_mode and fig_live is not None:
                im_topo.set_array(1 - rho_half_2d)
                ax_topo.set_title(f'Half-structure Topology - Iteration {opt_iter+1}', fontsize=12)
                
                iters = range(1, len(history_obj) + 1)
                line_conv.set_data(iters, history_obj)
                ax_conv.relim()
                ax_conv.autoscale_view()
                point_conv.set_offsets(np.array([[len(history_obj), history_obj[-1]]]))
                ax_conv.set_title(f'Compliance Convergence - Iteration {opt_iter+1}', fontsize=12)
                
                fig_live.canvas.draw_idle()
                fig_live.canvas.flush_events()
                plt.pause(live_pause_sec)
            
            # -- GIF frame capture (always happens regardless of live mode) --
            half_topo_frames.append(
                capture_topology_frame(
                    rho_half_2d,
                    f'Half-structure Topology - Iteration {opt_iter+1}',
                    [0, lx, 0, ly],
                    figsize=(12, 4)
                )
            )
            
            rho_full_2d = mirror_density(rho_half_2d.T).T
            full_topo_frames.append(
                capture_topology_frame(
                    rho_full_2d,
                    f'Full Structure Topology - Iteration {opt_iter+1}',
                    [0, 2*lx, 0, ly],
                    figsize=(16, 4)
                )
            )
            
            compliance_frames.append(
                capture_compliance_frame(history_obj, opt_iter + 1)
            )
        
        # Periodic checkpoint save every 20 iterations
        if (opt_iter + 1) % 20 == 0:
            checkpoint = {
                'iteration': opt_iter + 1,
                'rho': rho.detach().cpu().numpy(),
                'history_obj': np.array(history_obj),
                'history_vol': np.array(history_vol),
                'history_chg': np.array(history_chg),
                'history_loss': np.array(history_loss),
            }
            np.save(f'bridge_results/checkpoint_iter{opt_iter+1:03d}.npy', checkpoint)
            print(f"  -> Checkpoint saved at iteration {opt_iter+1}")
        
        # Early stopping criterion 1: small change
        if opt_iter > 80 and change < 0.003:
            print(f"Converged at iteration {opt_iter+1} (change < 0.003)")
            break
        
        # Early stopping criterion 2: compliance stable for last 30 iterations
        if opt_iter >= 100:
            recent_obj = np.array(history_obj[-30:])
            if np.std(recent_obj) / np.mean(recent_obj) < 0.02:  # Relative std
                print(f"Converged at iteration {opt_iter+1} (relative compliance std = {np.std(recent_obj)/np.mean(recent_obj):.4f} < 0.02)")
                break

    total_time = time.time() - start_time
    print(f"\nOptimization completed! Total time: {total_time:.2f} seconds")
    print(f"Final compliance: {history_obj[-1]:.4e}")
    print(f"Final volume fraction: {history_vol[-1]:.4f}")
    print(f"Lower boundary: diagonal (0,100)->(30,80), horizontal y=80, parabola (95,80)->(189,40)")
    print(f"Pier design domain: y∈[40,100] (base y∈[0,40] fixed)")
    print(f"Local symmetry: x∈[95,295] about pier centerline x=195")
    print(f"v3可视化 - Raised lower boundary horizontal to y=80, volfrac=0.50")

    # Save results
    rho_final = rho.view(nelx, nely).detach().cpu().numpy().T
    rho_binary = np.where(rho_final > 0.5, 1.0, 0.0)
    rho_full = mirror_density(rho_final.T)
    rho_full_binary = mirror_density(rho_binary.T)
    
    # Final topology plots (clean, no red overlays)
    fig_final, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    ax = axes[0, 0]
    ax.imshow(1 - rho_final, cmap='gray', origin='lower', extent=[0, lx, 0, ly], vmin=0, vmax=1)
    ax.set_title('(a) Half-structure Density', fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
    
    ax = axes[0, 1]
    ax.imshow(1 - rho_binary, cmap='gray', origin='lower', extent=[0, lx, 0, ly], vmin=0, vmax=1)
    ax.set_title('(b) Half-structure Binarized', fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
    
    ax = axes[1, 0]
    ax.imshow(1 - rho_full.T, cmap='gray', origin='lower', extent=[0, 2*lx, 0, ly], vmin=0, vmax=1)
    ax.set_title('(c) Full Structure Density', fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
    
    ax = axes[1, 1]
    ax.imshow(1 - rho_full_binary.T, cmap='gray', origin='lower', extent=[0, 2*lx, 0, ly], vmin=0, vmax=1)
    ax.set_title('(d) Full Structure Binarized', fontsize=12)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('bridge_results/final_topology.png', dpi=300, bbox_inches='tight')
    print("Saved: bridge_results/final_topology.png")
    
    # Compliance curve
    fig_conv, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(history_obj)+1), history_obj, 'b-', linewidth=2, label='Compliance')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Compliance', fontsize=12)
    ax.set_title('Compliance Convergence History', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('bridge_results/compliance_curve.png', dpi=300, bbox_inches='tight')
    print("Saved: bridge_results/compliance_curve.png")
    
    # Full Structure Topology — black & white
    fig_color, ax = plt.subplots(figsize=(16, 5))
    rgb_img = np.zeros((*rho_full_binary.T.shape, 3))
    gray = 1 - rho_full_binary.T    # 0 = solid (black), 1 = void (white)
    rgb_img[:, :, 0] = gray
    rgb_img[:, :, 1] = gray
    rgb_img[:, :, 2] = gray
    ax.imshow(rgb_img, origin='lower', extent=[0, 2*lx, 0, ly])
    ax.set_title('Full Structure Topology (Black=Solid, White=Void)', fontsize=14)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('bridge_results/full_structure_color.png', dpi=300, bbox_inches='tight')
    print("Saved: bridge_results/full_structure_color.png")
    
    # ==========================================
    # 13. Save Animated GIFs
    # ==========================================
    print("\n" + "="*60)
    print("Saving animated GIFs...")
    print("="*60)
    
    save_gif(
        half_topo_frames,
        'bridge_results/topology_half_animation.gif',
        duration_ms=gif_duration_ms,
        loop=0
    )
    
    save_gif(
        full_topo_frames,
        'bridge_results/topology_full_animation.gif',
        duration_ms=gif_duration_ms,
        loop=0
    )
    
    save_gif(
        compliance_frames,
        'bridge_results/compliance_animation.gif',
        duration_ms=gif_duration_ms,
        loop=0
    )
    
    # -----------------------------------------------------------------
    #  KEEP LIVE WINDOW OPEN AT THE END (if interactive)
    # -----------------------------------------------------------------
    if interactive_mode and fig_live is not None:
        # Final update with complete data
        rho_half_2d = rho.view(nelx, nely).detach().cpu().numpy().T
        im_topo.set_array(1 - rho_half_2d)
        ax_topo.set_title('Half-structure Topology - FINAL', fontsize=12)
        
        iters = range(1, len(history_obj) + 1)
        line_conv.set_data(iters, history_obj)
        ax_conv.relim()
        ax_conv.autoscale_view()
        point_conv.set_offsets(np.array([[len(history_obj), history_obj[-1]]]))
        ax_conv.set_title('Compliance Convergence - FINAL', fontsize=12)
        
        fig_live.canvas.draw_idle()
        fig_live.canvas.flush_events()
        print("\nLive display updated with final result.")
        print("Close the plot window to exit the program.")
        plt.ioff()          # turn off interactive mode
        plt.show()          # block until user closes the window
    
    # Save data
    np.save('bridge_results/rho_half.npy', rho_final)
    np.save('bridge_results/rho_half_binary.npy', rho_binary)
    np.save('bridge_results/rho_full.npy', rho_full.T)
    np.save('bridge_results/rho_full_binary.npy', rho_full_binary.T)
    np.save('bridge_results/history_obj.npy', np.array(history_obj))
    np.save('bridge_results/history_vol.npy', np.array(history_vol))
    np.save('bridge_results/history_chg.npy', np.array(history_chg))
    np.save('bridge_results/history_loss.npy', np.array(history_loss))
    print("Saved all .npy data files")
    
    print("\nAll results saved to 'bridge_results/' directory")
    return rho_final, rho_binary, rho_full, rho_full_binary, history_obj

if __name__ == '__main__':
    main()
