import torch
import numpy as np
import torch.nn as nn
import math

def assemble_element(node_list, element_list, dim=2):
    coord_list = []
    for tmp_nodes in element_list:
        tmp_coord = []
        for node_id in tmp_nodes:
            tmp_coord.append(node_list[node_id-1])
        if type(node_list) == type([]):
            coord_list.append(tmp_coord)
        else:
            coord_list.append(torch.cat(tmp_coord))
    if type(node_list) == type([]):
        return np.array(coord_list).swapaxes(1, 2) # (N, 2, 4)
    else:
        return torch.stack(coord_list, dim=0)  # (N, 2, 4)


class OptimFEM(nn.Module):
    def __init__(self, all_nodes, all_elements, material, bcs, dim=2, device='cuda'):
        super(OptimFEM, self).__init__()
        self.all_nodes = all_nodes
        self.all_elements = all_elements
        self.num_nodes = len(self.all_nodes)
        self.num_elements = len(self.all_elements)
        self.material_param = material
        self.material = torch.from_numpy(self.__init_material__(material).astype('float32')).to(device).unsqueeze(0).repeat_interleave(self.num_elements, dim=0)
        self.device = device
        self.dim = dim
        # self.disp_vect = nn.Parameter(torch.randn(self.num_nodes,  self.dim, requires_grad=True, dtype=torch.float32).to(device))
        self.disp_vect = nn.ParameterList([nn.Parameter(torch.randn(1).to(device)*0.1) for _ in range(self.num_nodes * 2)])
        self.disp_array_element = assemble_element(self.disp_vect, self.all_elements)
        # self.disp_bc, self.disp_bc_mask = self.__init_bcs__(bcs)
        self.bcs = bcs
        self.weight = 1  # weight of boundary loss to total strain energy
        self.gauss_points_xi = [-1/np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)]
        self.gauss_points_eta = [-1/np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]
        self.gauss_weights = [1, 1, 1, 1]
        self.num_gauss_point = 4
        self.element_coord_list = assemble_element(self.all_nodes, self.all_elements)
        self.__init_ele_inv__()
        # self.jacobi, self.jacobi_inv = self.__init_calc_jacobi__()
        # self.B = torch.from_numpy(self.__init_calc_strainform__().astype('float32')).to(device)
        self.debug = False

    def __init_material__(self, material):
        if material['type'] == 'plane_stress':
            E = material['E']
            nu = material['nu']
            D = (E/(1-nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
        else:
            D=None
        return D
    #
    def __init_ele_inv__(self):
        x1 = self.element_coord_list[:, 0, 0]
        x2 = self.element_coord_list[:, 0, 1] - x1
        x3 = self.element_coord_list[:, 0, 2] - x1
        x4 = self.element_coord_list[:, 0, 3] - x1
        y1 = self.element_coord_list[:, 1, 0]
        y2 = self.element_coord_list[:, 1, 1] - y1
        y3 = self.element_coord_list[:, 1, 2] - y1
        y4 = self.element_coord_list[:, 1, 3] - y1

        inv = 1 / (-x2*x3*y3*y4 + x2*x4*y4*y3 + x2*y2*x3*y4 - x2*y2*x4*y3 - x3*x4*y4*y2 + x3*y3*x4*y2)
        a_1 = (-x3*y3*y4 + x4*y4*y3) * inv
        a_2 = (x2*y2*y4 - x4*y4*y2) * inv
        a_3 = (-x2*y2*y3 + x3*y3*y2) * inv
        b_1 = (-x3*x4*y4 + x3*y3*x4) * inv
        b_2 = (x2*x4*y4 - x2*y2*x4) * inv
        b_3 = (-x2*x3*y3 + x2*y2*x3) * inv
        c_1 = (x3*y4 - x4*y3) * inv
        c_2 = (-x2*y4 + x4*y2) * inv
        c_3 = (x2*y3 - x3*y2) * inv

        p_1 = (1 - 1 / math.sqrt(3)) / 2
        p_2 = (1 + 1 / math.sqrt(3)) / 2
        inter_x1 = p_1 * ((p_1 * (x3 - x4) + x4) - (p_1 * (x2 - 0.0) + 0.0)) + (p_1 * (x2 - 0.0) + 0.0)
        inter_x2 = p_1 * ((p_2 * (x3 - x4) + x4) - (p_2 * (x2 - 0.0) + 0.0)) + (p_2 * (x2 - 0.0) + 0.0)
        inter_x3 = p_2 * ((p_2 * (x3 - x4) + x4) - (p_2 * (x2 - 0.0) + 0.0)) + (p_2 * (x2 - 0.0) + 0.0)
        inter_x4 = p_2 * ((p_1 * (x3 - x4) + x4) - (p_1 * (x2 - 0.0) + 0.0)) + (p_1 * (x2 - 0.0) + 0.0)
        inter_y1 = p_1 * ((p_1 * (y3 - y4) + y4) - (p_1 * (y2 - 0.0) + 0.0)) + (p_1 * (y2 - 0.0) + 0.0)
        inter_y2 = p_1 * ((p_2 * (y3 - y4) + y4) - (p_2 * (y2 - 0.0) + 0.0)) + (p_2 * (y2 - 0.0) + 0.0)
        inter_y3 = p_2 * ((p_2 * (y3 - y4) + y4) - (p_2 * (y2 - 0.0) + 0.0)) + (p_2 * (y2 - 0.0) + 0.0)
        inter_y4 = p_2 * ((p_1 * (y3 - y4) + y4) - (p_1 * (y2 - 0.0) + 0.0)) + (p_1 * (y2 - 0.0) + 0.0)
        self.weight_geo = [torch.from_numpy(t.astype('float32')).to(self.device) for t in [a_1, a_2, a_3, b_1, b_2, b_3, c_1, c_2, c_3]]
        self.inter_points = [torch.from_numpy(t.astype('float32')).to(self.device) for t in [inter_x1, inter_y1, inter_x2, inter_y2, inter_x3, inter_y3, inter_x4, inter_y4]]

    def __calc_element_energy__(self):
        energy = 0.0
        E, nu = self.material_param['E'], self.material_param['nu']
        for i in range(self.num_elements):  # 逐个单元计算应变能然后叠加
            tmp_element_into = self.all_elements[i]
            epsilon_xx, epsilon_yy, gamma_xy = 0.0, 0.0, 0.0
            u_1 = self.disp_vect[(tmp_element_into[0]-1) * 2]
            u_2 = self.disp_vect[(tmp_element_into[1]-1) * 2]
            u_3 = self.disp_vect[(tmp_element_into[2]-1) * 2]
            u_4 = self.disp_vect[(tmp_element_into[3]-1) * 2]
            v_1 = self.disp_vect[(tmp_element_into[0]-1) * 2 + 1]
            v_2 = self.disp_vect[(tmp_element_into[1]-1) * 2 + 1]
            v_3 = self.disp_vect[(tmp_element_into[2]-1) * 2 + 1]
            v_4 = self.disp_vect[(tmp_element_into[3]-1) * 2 + 1]
            a_u = self.weight_geo[0][i] * (u_2 - u_1) + self.weight_geo[1][i] * (u_3 - u_1) + self.weight_geo[2][i] * (u_4 - u_1)
            b_u = self.weight_geo[3][i] * (u_2 - u_1) + self.weight_geo[4][i] * (u_3 - u_1) + self.weight_geo[5][i] * (u_4 - u_1)
            c_u = self.weight_geo[6][i] * (u_2 - u_1) + self.weight_geo[7][i] * (u_3 - u_1) + self.weight_geo[8][i] * (u_4 - u_1)
            a_v = self.weight_geo[0][i] * (v_2 - v_1) + self.weight_geo[1][i] * (v_3 - v_1) + self.weight_geo[2][i] * (v_4 - v_1)
            b_v = self.weight_geo[3][i] * (v_2 - v_1) + self.weight_geo[4][i] * (v_3 - v_1) + self.weight_geo[5][i] * (v_4 - v_1)
            c_v = self.weight_geo[6][i] * (v_2 - v_1) + self.weight_geo[7][i] * (v_3 - v_1) + self.weight_geo[8][i] * (v_4 - v_1)
            for j in range(4):
                x_int = self.inter_points[2 * j][i]
                y_int = self.inter_points[2 * j + 1][i]
                epsilon_xx += a_u + c_u * y_int
                gamma_xy += b_u + c_u * x_int + a_v + c_v * y_int
                epsilon_yy += b_v + c_v * x_int
            sigma_xx = E / (1 - nu ** 2) * (epsilon_xx + nu * epsilon_yy)
            sigma_yy = E / (1 - nu ** 2) * (epsilon_yy + nu * epsilon_xx)
            sigma_xy = E / (1 + nu) * gamma_xy
            tmp_energy = 0.5 * (epsilon_xx * sigma_xx + epsilon_yy * sigma_yy + 2 * gamma_xy * sigma_xy)
            if self.debug:
                print(i, tmp_energy.item())
            energy += tmp_energy**2
        return energy

    def __calc_bc_loss__(self):
        loss = 0.0
        for bc_set in self.bcs:
            tmp_node_id, tmp_opt, tmp_bc = bc_set
            if tmp_opt == 'u':
                for i in range(len(tmp_node_id)):
                    loss += (self.disp_vect[(tmp_node_id[i]-1)*2] - tmp_bc[i]) ** 2
            elif tmp_opt == 'v':
                for i in range(len(tmp_node_id)):
                    loss += (self.disp_vect[(tmp_node_id[i]-1)*2+1] - tmp_bc[i]) ** 2
        return loss

    def forward(self, x=None):
        total_energy = self.__calc_element_energy__()
        bc_constrain = self.__calc_bc_loss__()
        return total_energy + bc_constrain * total_energy.item()


class OptimFEM_inverse(nn.Module):
    def __init__(self, all_nodes, all_elements, material, bcs, dim=2, device='cuda'):
        super(OptimFEM_inverse, self).__init__()
        self.all_nodes = all_nodes
        self.all_elements = all_elements
        self.num_nodes = len(self.all_nodes)
        self.num_elements = len(self.all_elements)
        self.material_param = material
        self.material = torch.from_numpy(self.__init_material__(material).astype('float32')).to(device).unsqueeze(0).repeat_interleave(self.num_elements, dim=0)
        self.device = device
        self.dim = dim
        # self.disp_vect = nn.Parameter(torch.randn(self.num_nodes,  self.dim, requires_grad=True, dtype=torch.float32).to(device))
        self.disp_vect = nn.ParameterList([nn.Parameter(torch.randn(1).to(device)*0.1) for _ in range(self.num_nodes * 2)])
        self.disp_array_element = assemble_element(self.disp_vect, self.all_elements)
        # self.disp_bc, self.disp_bc_mask = self.__init_bcs__(bcs)
        self.bcs = bcs
        self.weight = 1  # weight of boundary loss to total strain energy
        self.gauss_points_xi = [-1/np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)]
        self.gauss_points_eta = [-1/np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]
        self.gauss_weights = [1, 1, 1, 1]
        self.num_gauss_point = 4
        self.element_coord_list = assemble_element(self.all_nodes, self.all_elements)
        self.__init_ele_inv__()
        # self.jacobi, self.jacobi_inv = self.__init_calc_jacobi__()
        # self.B = torch.from_numpy(self.__init_calc_strainform__().astype('float32')).to(device)
        self.debug = False

    def __init_material__(self, material):
        if material['type'] == 'plane_stress':
            E = material['E']
            nu = material['nu']
            D = (E/(1-nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
        else:
            D=None
        return D
    #
    def __init_ele_inv__(self):
        x1 = self.element_coord_list[:, 0, 0]
        x2 = self.element_coord_list[:, 0, 1] - x1
        x3 = self.element_coord_list[:, 0, 2] - x1
        x4 = self.element_coord_list[:, 0, 3] - x1
        y1 = self.element_coord_list[:, 1, 0]
        y2 = self.element_coord_list[:, 1, 1] - y1
        y3 = self.element_coord_list[:, 1, 2] - y1
        y4 = self.element_coord_list[:, 1, 3] - y1

        inv = 1 / (-x2*x3*y3*y4 + x2*x4*y4*y3 + x2*y2*x3*y4 - x2*y2*x4*y3 - x3*x4*y4*y2 + x3*y3*x4*y2)
        a_1 = (-x3*y3*y4 + x4*y4*y3) * inv
        a_2 = (x2*y2*y4 - x4*y4*y2) * inv
        a_3 = (-x2*y2*y3 + x3*y3*y2) * inv
        b_1 = (-x3*x4*y4 + x3*y3*x4) * inv
        b_2 = (x2*x4*y4 - x2*y2*x4) * inv
        b_3 = (-x2*x3*y3 + x2*y2*x3) * inv
        c_1 = (x3*y4 - x4*y3) * inv
        c_2 = (-x2*y4 + x4*y2) * inv
        c_3 = (x2*y3 - x3*y2) * inv

        p_1 = (1 - 1 / math.sqrt(3)) / 2
        p_2 = (1 + 1 / math.sqrt(3)) / 2
        inter_x1 = p_1 * ((p_1 * (x3 - x4) + x4) - (p_1 * (x2 - 0.0) + 0.0)) + (p_1 * (x2 - 0.0) + 0.0)
        inter_x2 = p_1 * ((p_2 * (x3 - x4) + x4) - (p_2 * (x2 - 0.0) + 0.0)) + (p_2 * (x2 - 0.0) + 0.0)
        inter_x3 = p_2 * ((p_2 * (x3 - x4) + x4) - (p_2 * (x2 - 0.0) + 0.0)) + (p_2 * (x2 - 0.0) + 0.0)
        inter_x4 = p_2 * ((p_1 * (x3 - x4) + x4) - (p_1 * (x2 - 0.0) + 0.0)) + (p_1 * (x2 - 0.0) + 0.0)
        inter_y1 = p_1 * ((p_1 * (y3 - y4) + y4) - (p_1 * (y2 - 0.0) + 0.0)) + (p_1 * (y2 - 0.0) + 0.0)
        inter_y2 = p_1 * ((p_2 * (y3 - y4) + y4) - (p_2 * (y2 - 0.0) + 0.0)) + (p_2 * (y2 - 0.0) + 0.0)
        inter_y3 = p_2 * ((p_2 * (y3 - y4) + y4) - (p_2 * (y2 - 0.0) + 0.0)) + (p_2 * (y2 - 0.0) + 0.0)
        inter_y4 = p_2 * ((p_1 * (y3 - y4) + y4) - (p_1 * (y2 - 0.0) + 0.0)) + (p_1 * (y2 - 0.0) + 0.0)
        self.weight_geo = [torch.from_numpy(t.astype('float32')).to(self.device) for t in [a_1, a_2, a_3, b_1, b_2, b_3, c_1, c_2, c_3]]
        self.inter_points = [torch.from_numpy(t.astype('float32')).to(self.device) for t in [inter_x1, inter_y1, inter_x2, inter_y2, inter_x3, inter_y3, inter_x4, inter_y4]]

    def __calc_element_energy__(self):
        energy = 0.0
        E, nu = self.material_param['E'], self.material_param['nu']
        for i in range(self.num_elements):  # 逐个单元计算应变能然后叠加
            tmp_element_into = self.all_elements[i]
            epsilon_xx, epsilon_yy, gamma_xy = 0.0, 0.0, 0.0
            u_1 = self.disp_vect[(tmp_element_into[0]-1) * 2]
            u_2 = self.disp_vect[(tmp_element_into[1]-1) * 2]
            u_3 = self.disp_vect[(tmp_element_into[2]-1) * 2]
            u_4 = self.disp_vect[(tmp_element_into[3]-1) * 2]
            v_1 = self.disp_vect[(tmp_element_into[0]-1) * 2 + 1]
            v_2 = self.disp_vect[(tmp_element_into[1]-1) * 2 + 1]
            v_3 = self.disp_vect[(tmp_element_into[2]-1) * 2 + 1]
            v_4 = self.disp_vect[(tmp_element_into[3]-1) * 2 + 1]
            a_u = self.weight_geo[0][i] * (u_2 - u_1) + self.weight_geo[1][i] * (u_3 - u_1) + self.weight_geo[2][i] * (u_4 - u_1)
            b_u = self.weight_geo[3][i] * (u_2 - u_1) + self.weight_geo[4][i] * (u_3 - u_1) + self.weight_geo[5][i] * (u_4 - u_1)
            c_u = self.weight_geo[6][i] * (u_2 - u_1) + self.weight_geo[7][i] * (u_3 - u_1) + self.weight_geo[8][i] * (u_4 - u_1)
            a_v = self.weight_geo[0][i] * (v_2 - v_1) + self.weight_geo[1][i] * (v_3 - v_1) + self.weight_geo[2][i] * (v_4 - v_1)
            b_v = self.weight_geo[3][i] * (v_2 - v_1) + self.weight_geo[4][i] * (v_3 - v_1) + self.weight_geo[5][i] * (v_4 - v_1)
            c_v = self.weight_geo[6][i] * (v_2 - v_1) + self.weight_geo[7][i] * (v_3 - v_1) + self.weight_geo[8][i] * (v_4 - v_1)
            for j in range(4):
                x_int = self.inter_points[2 * j][i]
                y_int = self.inter_points[2 * j + 1][i]
                epsilon_xx += a_u + c_u * y_int
                gamma_xy += b_u + c_u * x_int + a_v + c_v * y_int
                epsilon_yy += b_v + c_v * x_int
            sigma_xx = E / (1 - nu ** 2) * (epsilon_xx + nu * epsilon_yy)
            sigma_yy = E / (1 - nu ** 2) * (epsilon_yy + nu * epsilon_xx)
            sigma_xy = E / (1 + nu) * gamma_xy
            tmp_energy = 0.5 * (epsilon_xx * sigma_xx + epsilon_yy * sigma_yy + 2 * gamma_xy * sigma_xy)
            if self.debug:
                print(i, tmp_energy.item())
            energy += tmp_energy**2
        return energy

    def __calc_bc_loss__(self):
        loss = 0.0
        for bc_set in self.bcs:
            tmp_node_id, tmp_opt, tmp_bc = bc_set
            if tmp_opt == 'u':
                for i in range(len(tmp_node_id)):
                    loss += (self.disp_vect[(tmp_node_id[i]-1)*2] - tmp_bc[i]) ** 2
            elif tmp_opt == 'v':
                for i in range(len(tmp_node_id)):
                    loss += (self.disp_vect[(tmp_node_id[i]-1)*2+1] - tmp_bc[i]) ** 2
        return loss

    def forward(self, x=None):
        total_energy = self.__calc_element_energy__()
        bc_constrain = self.__calc_bc_loss__()
        return total_energy + bc_constrain * total_energy.item()


class OptimFEM_FreeMesh2D(nn.Module):
    def __init__(self, all_nodes, all_elements, material, bcs, dim=2, device='cuda'):
        super(OptimFEM_FreeMesh2D, self).__init__()
        self.all_nodes = all_nodes
        self.all_nodes_elesize = # 节点所在单元最小尺寸，用于初始化位移
        self.all_elements_quad = all_elements['C2D4']
        self.all_elements_tri = all_elements['C2D3']
        self.num_nodes = len(self.all_nodes)
        self.num_elements_quad = len(self.all_elements_quad)
        self.num_elements_tri = len(self.all_elements_tri)
        self.material_param = material
        self.material = torch.from_numpy(self.__init_material__(material).astype('float32')).to(device).unsqueeze(0).repeat_interleave(self.num_elements, dim=0)
        self.device = device
        self.dim = dim
        # self.disp_vect = nn.Parameter(torch.randn(self.num_nodes,  self.dim, requires_grad=True, dtype=torch.float32).to(device))
        self.disp_vect = nn.ParameterList([nn.Parameter(torch.randn(1).to(device)*0.1) for _ in range(self.num_nodes * 2)])
        # self.disp_bc, self.disp_bc_mask = self.__init_bcs__(bcs)
        self.bcs = bcs
        self.weight = 1  # weight of boundary loss to total strain energy
        self.gauss_points_xi = [-1/np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)]
        self.gauss_points_eta = [-1/np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]
        self.gauss_weights = [1, 1, 1, 1]
        self.num_gauss_point = 4
        self.element_coord_list = assemble_element(self.all_nodes, self.all_elements)
        self.__init_ele_inv__()
        # self.jacobi, self.jacobi_inv = self.__init_calc_jacobi__()
        # self.B = torch.from_numpy(self.__init_calc_strainform__().astype('float32')).to(device)
        self.debug = False

    def __init_material__(self, material):
        if material['type'] == 'plane_stress':
            E = material['E']
            nu = material['nu']
            D = (E/(1-nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
        else:
            D=None
        return D
    #
    def __init_ele_inv__(self):
        x1 = self.element_coord_list[:, 0, 0]
        x2 = self.element_coord_list[:, 0, 1] - x1
        x3 = self.element_coord_list[:, 0, 2] - x1
        x4 = self.element_coord_list[:, 0, 3] - x1
        y1 = self.element_coord_list[:, 1, 0]
        y2 = self.element_coord_list[:, 1, 1] - y1
        y3 = self.element_coord_list[:, 1, 2] - y1
        y4 = self.element_coord_list[:, 1, 3] - y1

        inv = 1 / (-x2*x3*y3*y4 + x2*x4*y4*y3 + x2*y2*x3*y4 - x2*y2*x4*y3 - x3*x4*y4*y2 + x3*y3*x4*y2)
        a_1 = (-x3*y3*y4 + x4*y4*y3) * inv
        a_2 = (x2*y2*y4 - x4*y4*y2) * inv
        a_3 = (-x2*y2*y3 + x3*y3*y2) * inv
        b_1 = (-x3*x4*y4 + x3*y3*x4) * inv
        b_2 = (x2*x4*y4 - x2*y2*x4) * inv
        b_3 = (-x2*x3*y3 + x2*y2*x3) * inv
        c_1 = (x3*y4 - x4*y3) * inv
        c_2 = (-x2*y4 + x4*y2) * inv
        c_3 = (x2*y3 - x3*y2) * inv

        p_1 = (1 - 1 / math.sqrt(3)) / 2
        p_2 = (1 + 1 / math.sqrt(3)) / 2
        inter_x1 = p_1 * ((p_1 * (x3 - x4) + x4) - (p_1 * (x2 - 0.0) + 0.0)) + (p_1 * (x2 - 0.0) + 0.0)
        inter_x2 = p_1 * ((p_2 * (x3 - x4) + x4) - (p_2 * (x2 - 0.0) + 0.0)) + (p_2 * (x2 - 0.0) + 0.0)
        inter_x3 = p_2 * ((p_2 * (x3 - x4) + x4) - (p_2 * (x2 - 0.0) + 0.0)) + (p_2 * (x2 - 0.0) + 0.0)
        inter_x4 = p_2 * ((p_1 * (x3 - x4) + x4) - (p_1 * (x2 - 0.0) + 0.0)) + (p_1 * (x2 - 0.0) + 0.0)
        inter_y1 = p_1 * ((p_1 * (y3 - y4) + y4) - (p_1 * (y2 - 0.0) + 0.0)) + (p_1 * (y2 - 0.0) + 0.0)
        inter_y2 = p_1 * ((p_2 * (y3 - y4) + y4) - (p_2 * (y2 - 0.0) + 0.0)) + (p_2 * (y2 - 0.0) + 0.0)
        inter_y3 = p_2 * ((p_2 * (y3 - y4) + y4) - (p_2 * (y2 - 0.0) + 0.0)) + (p_2 * (y2 - 0.0) + 0.0)
        inter_y4 = p_2 * ((p_1 * (y3 - y4) + y4) - (p_1 * (y2 - 0.0) + 0.0)) + (p_1 * (y2 - 0.0) + 0.0)
        self.weight_geo = [torch.from_numpy(t.astype('float32')).to(self.device) for t in [a_1, a_2, a_3, b_1, b_2, b_3, c_1, c_2, c_3]]
        self.inter_points = [torch.from_numpy(t.astype('float32')).to(self.device) for t in [inter_x1, inter_y1, inter_x2, inter_y2, inter_x3, inter_y3, inter_x4, inter_y4]]

    def __calc_element_energy__(self):
        energy = 0.0
        E, nu = self.material_param['E'], self.material_param['nu']
        for i in range(self.num_elements):  # 逐个单元计算应变能然后叠加
            tmp_element_into = self.all_elements[i]
            epsilon_xx, epsilon_yy, gamma_xy = 0.0, 0.0, 0.0
            u_1 = self.disp_vect[(tmp_element_into[0]-1) * 2]
            u_2 = self.disp_vect[(tmp_element_into[1]-1) * 2]
            u_3 = self.disp_vect[(tmp_element_into[2]-1) * 2]
            u_4 = self.disp_vect[(tmp_element_into[3]-1) * 2]
            v_1 = self.disp_vect[(tmp_element_into[0]-1) * 2 + 1]
            v_2 = self.disp_vect[(tmp_element_into[1]-1) * 2 + 1]
            v_3 = self.disp_vect[(tmp_element_into[2]-1) * 2 + 1]
            v_4 = self.disp_vect[(tmp_element_into[3]-1) * 2 + 1]
            a_u = self.weight_geo[0][i] * (u_2 - u_1) + self.weight_geo[1][i] * (u_3 - u_1) + self.weight_geo[2][i] * (u_4 - u_1)
            b_u = self.weight_geo[3][i] * (u_2 - u_1) + self.weight_geo[4][i] * (u_3 - u_1) + self.weight_geo[5][i] * (u_4 - u_1)
            c_u = self.weight_geo[6][i] * (u_2 - u_1) + self.weight_geo[7][i] * (u_3 - u_1) + self.weight_geo[8][i] * (u_4 - u_1)
            a_v = self.weight_geo[0][i] * (v_2 - v_1) + self.weight_geo[1][i] * (v_3 - v_1) + self.weight_geo[2][i] * (v_4 - v_1)
            b_v = self.weight_geo[3][i] * (v_2 - v_1) + self.weight_geo[4][i] * (v_3 - v_1) + self.weight_geo[5][i] * (v_4 - v_1)
            c_v = self.weight_geo[6][i] * (v_2 - v_1) + self.weight_geo[7][i] * (v_3 - v_1) + self.weight_geo[8][i] * (v_4 - v_1)
            for j in range(4):
                x_int = self.inter_points[2 * j][i]
                y_int = self.inter_points[2 * j + 1][i]
                epsilon_xx += a_u + c_u * y_int
                gamma_xy += b_u + c_u * x_int + a_v + c_v * y_int
                epsilon_yy += b_v + c_v * x_int
            sigma_xx = E / (1 - nu ** 2) * (epsilon_xx + nu * epsilon_yy)
            sigma_yy = E / (1 - nu ** 2) * (epsilon_yy + nu * epsilon_xx)
            sigma_xy = E / (1 + nu) * gamma_xy
            tmp_energy = 0.5 * (epsilon_xx * sigma_xx + epsilon_yy * sigma_yy + 2 * gamma_xy * sigma_xy)
            if self.debug:
                print(i, tmp_energy.item())
            energy += tmp_energy**2
        return energy

    def __calc_bc_loss__(self):
        loss = 0.0
        for bc_set in self.bcs:
            tmp_node_id, tmp_opt, tmp_bc = bc_set
            if tmp_opt == 'u':
                for i in range(len(tmp_node_id)):
                    loss += (self.disp_vect[(tmp_node_id[i]-1)*2] - tmp_bc[i]) ** 2
            elif tmp_opt == 'v':
                for i in range(len(tmp_node_id)):
                    loss += (self.disp_vect[(tmp_node_id[i]-1)*2+1] - tmp_bc[i]) ** 2
        return loss

    def forward(self, x=None):
        total_energy = self.__calc_element_energy__()
        bc_constrain = self.__calc_bc_loss__()
        return total_energy + bc_constrain * total_energy.item()