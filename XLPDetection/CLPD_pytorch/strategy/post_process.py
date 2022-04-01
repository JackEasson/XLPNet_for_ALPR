import torch
from utils.gaussian_utils import GaussianNMS
from utils.general import corner2bbox_multi


# for one image
def clp_post_process(out_score_stage, out_corner_stage, conf_thres=0.3):
    device = out_score_stage[0].device
    corner_index_list = []
    for idx, (score_map, corner_map) in enumerate(zip(out_score_stage, out_corner_stage)):
        mask = score_map.mean(-1) >= conf_thres  # size(h, w)
        corner_map_candidate = corner_map[mask]  # size(n, 8)
        n = corner_map_candidate.shape[0]
        stage_index_map = torch.full((n, 1), idx, dtype=torch.float32, device=device)
        corner_index_map2d = torch.cat([corner_map_candidate, stage_index_map], dim=-1)  # size(n, 9)
        corner_index_list.append(corner_index_map2d)
    corner_index_out = torch.cat(corner_index_list, dim=-1)  # size(n1+n2, 9)


class CLPPostProcessor:
    def __init__(self, conf_thres, gauss_ratio=2., device='cpu', nms_thres=0.3, max_obj=50, use_distance=False):
        self.conf_thres = conf_thres
        self.device = device
        self.max_obj = max_obj
        self.gauss_nms = GaussianNMS(gauss_ratio, device, nms_thres, use_distance)

    def __call__(self, out_score_map, out_corner_map):
        score_map = out_score_map.mean(-1)
        score_flatten = score_map.reshape(-1)
        topk_scores, indices = torch.topk(score_flatten, self.max_obj, largest=True, sorted=False)
        topk_corners = out_corner_map.reshape(-1, 8)[indices]
        mask = topk_scores >= self.conf_thres  # size(n)
        scores_candidate = topk_scores[mask]  # size(n)
        corners_candidate = topk_corners[mask]  # size(n, 8)
        if scores_candidate.size(0) == 0:
            return None
        corner_score_map2d = torch.cat([scores_candidate.unsqueeze(-1), corners_candidate], dim=-1)  # size(n, 9)
        keep_index = self.gauss_nms(corner_score_map2d[:, 1:9], corner_score_map2d[:, 0])
        nms_out = corner_score_map2d[keep_index]  # size(n', 10) last_dim->(score_map_candidate, corner_map_candidate)
        return nms_out


def nms_out_corners2bboxes(nms_out):
    """
    :param nms_out: size(n, 9)
    :return:
    """
    bbox_out = corner2bbox_multi(nms_out[:, 1:9])  # size(n, 4)
    score_out = nms_out[:, 0].unsqueeze(-1)
    score_bbox_out = torch.cat([score_out, bbox_out], dim=-1)  # size(n, 5)
    return score_bbox_out

