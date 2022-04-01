import torch


# corner to reg target
def corner_encode(corner, expand_grids_map, stage_lvl=4, S0=16):
    """
    :param corner: size(8)
    :param expand_grids_map: size(h, w, 8) last dim: xyxyxyxy
    :param stage_lvl:
    :param S0: 16 from fovea box
    :return: size(h, w, 8)
    """
    zeta = (4 ** stage_lvl * S0) ** 0.5
    reg_map = expand_grids_map + 0.5
    # grids_center_maps = grids_center_maps.unsqueeze(2).repeat(1, 1, 4, 1)  # size(h, w, 1, 2)
    # reg_map = grids_center_maps.reshape(h, w, -1)  # size(h, w, 8)
    # x
    reg_x_map = (corner[0::2] - 2 ** stage_lvl * reg_map[..., 0::2]) / zeta
    reg_x_map_pow = reg_x_map.abs() ** (1 / 3)
    reg_x_map_pow[reg_x_map < 0] *= -1
    reg_map[..., 0::2] = reg_x_map_pow
    # y
    reg_y_map = (corner[1::2] - 2 ** stage_lvl * reg_map[..., 1::2]) / zeta
    reg_y_map_pow = reg_y_map.abs() ** (1 / 3)
    reg_y_map_pow[reg_y_map < 0] *= -1
    reg_map[..., 1::2] = reg_y_map_pow
    return reg_map


def corner_decode(reg_map, expand_grids_map, stage_lvl=4, S0=16):
    """
    :param reg_map: size(h, w, 8) or size(b, h, w, 8)
    :param expand_grids_map: size(h, w, 8) last dim: xyxyxyxy
    :param stage_lvl:
    :param S0:
    :return:
    """
    zeta = (4 ** stage_lvl * S0) ** 0.5
    grids_center_map = expand_grids_map + 0.5
    corners_map = reg_map ** 3 * zeta + grids_center_map * (2 ** stage_lvl)
    return corners_map


def corner_decode_onnx_version(reg_map, expand_grids_map, stage_lvl=4, S0=16):
    """
    :param reg_map: size(h, w, 8) or size(b, h, w, 8)
    :param expand_grids_map: size(h, w, 8) last dim: xyxyxyxy
    :param stage_lvl:
    :param S0:
    :return:
    """
    zeta = (4 ** stage_lvl * S0) ** 0.5
    grids_center_map = expand_grids_map + 0.5
    corners_map = reg_map * reg_map * reg_map * zeta + grids_center_map * (2 ** stage_lvl)
    return corners_map
