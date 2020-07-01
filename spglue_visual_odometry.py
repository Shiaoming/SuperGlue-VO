import numpy as np
import torch
import cv2

from models.matching import Matching
from sp_extractor import PointTracker

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
    def __init__(self, cam, annotations):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.cur_Ro = None
        self.cur_to = None
        self.px_refo = None
        self.px_curo = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0

        # This code is from https://github.com/magicleap/SuperGluePretrainedNetwork
        nms_dist = 4
        conf_thresh = 0.015
        sinkhorn_iterations = 20
        self.nn_thresh = 0.7
        match_threshold = 0.6
        cuda = False
        config = {
            'superpoint': {
                'nms_radius': nms_dist,
                'keypoint_threshold': conf_thresh,
                'max_keypoints': -1
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }

        self.device = 'cuda' if torch.cuda.is_available() and cuda else 'cpu'

        self.keys = ['keypoints', 'scores', 'descriptors']
        self.matching = Matching(config).eval().to(self.device)

        self.tracker = PointTracker(
            max_length=2, nn_thresh=self.nn_thresh)

        with open(annotations) as f:
            self.annotations = f.readlines()

    def featureTracking0(self, pts, desc):
        # Add points and descriptors to the tracker.
        self.tracker.update(pts, desc)
        # Get tracks for points which were match successfully across all frames.
        tracks = self.tracker.get_tracks(min_length=1)
        # Normalize track scores to [0,1].
        tracks[:, 1] /= float(self.nn_thresh)
        kp1, kp2 = self.tracker.draw_tracks(tracks)
        return kp1, kp2

    def featureTracking(self):
        # This code is from https://github.com/magicleap/SuperGluePretrainedNetwork
        self.new_frame = torch.from_numpy(self.new_frame / 255.).float()[None, None].to(self.device)
        pred = self.matching({**self.last_data, 'image1': self.new_frame})
        kpts0 = self.last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()

        valid = matches > -1
        kp1 = kpts0[valid]
        kp2 = kpts1[matches[valid]]

        desc = pred['descriptors1'][0].cpu().detach().numpy()
        pts = np.zeros((3, kpts1.shape[0]))  # Populate point data sized 3xN.
        pts[0, :] = kpts1[:, 0]
        pts[1, :] = kpts1[:, 1]
        pts[2, :] = pred['scores1'][0].cpu().detach().numpy()
        kp1o, kp2o = self.featureTracking0(pts, desc)

        return kp1, kp2, kp1o, kp2o

    def getAbsoluteScale(self, frame_id):  # specialized for KITTI odometry dataset
        ss = self.annotations[frame_id - 1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))

    def processFirstFrame(self):
        # This code is from https://github.com/magicleap/SuperGluePretrainedNetwork
        self.new_frame = torch.from_numpy(self.new_frame / 255.).float()[None, None].to(self.device)
        self.last_data = self.matching.superpoint({'image': self.new_frame})
        self.last_data = {k + '0': self.last_data[k] for k in self.keys}
        self.last_data['image0'] = self.new_frame
        self.px_ref = self.last_data['keypoints0']
        self.px_refo = self.last_data['keypoints0'][0].cpu().numpy()
        self.frame_stage = STAGE_SECOND_FRAME

        desc = self.last_data['descriptors0'][0].cpu().detach().numpy()
        pts = np.zeros((3, self.px_refo.shape[0]))  # Populate point data sized 3xN.
        pts[0, :] = self.px_refo[:, 0]
        pts[1, :] = self.px_refo[:, 1]
        pts[2, :] = self.last_data['scores0'][0].cpu().detach().numpy()
        self.featureTracking0(pts, desc)

    def processSecondFrame(self):
        self.px_ref, self.px_cur, self.px_refo, self.px_curo = self.featureTracking()

        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,
                                       focal=self.focal, pp=self.pp,
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
                                                          focal=self.focal, pp=self.pp)
        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur

        # SuperPoint tracker
        E, mask = cv2.findEssentialMat(self.px_curo, self.px_refo,
                                       focal=self.focal, pp=self.pp,
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.cur_Ro, self.cur_to, mask = cv2.recoverPose(E, self.px_curo, self.px_refo,
                                                            focal=self.focal, pp=self.pp)
        self.px_refo = self.px_curo

    def processFrame(self, frame_id):
        self.px_ref, self.px_cur, self.px_refo, self.px_curo = self.featureTracking()

        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,
                                       focal=self.focal, pp=self.pp,
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
                                        focal=self.focal, pp=self.pp)
        absolute_scale = self.getAbsoluteScale(frame_id)
        if (absolute_scale > 0.1):
            self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)
        self.px_ref = self.px_cur

        # SuperPoint tracker
        E, mask = cv2.findEssentialMat(self.px_curo, self.px_refo,
                                       focal=self.focal, pp=self.pp,
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, self.px_curo, self.px_refo,
                                        focal=self.focal, pp=self.pp)
        if (absolute_scale > 0.1):
            self.cur_to = self.cur_to + absolute_scale * self.cur_Ro.dot(t)
            self.cur_Ro = R.dot(self.cur_Ro)
        self.px_refo = self.px_curo

    def update(self, img, frame_id):
        assert (img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[1] ==
                self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.new_frame = img
        if (self.frame_stage == STAGE_DEFAULT_FRAME):
            self.processFrame(frame_id)
        elif (self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame()
        elif (self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame()
        self.last_frame = self.new_frame
