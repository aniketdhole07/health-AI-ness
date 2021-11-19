#!/usr/bin/env python3

import cv2
import numpy as np
import open3d as o3d
import numpy as np


# From : https://stackoverflow.com/a/59026582/8574085

def calculate_zy_rotation_for_arrow(v):
    """
    Calculates the rotations required to go from the vector v to the 
    z axis vector. The first rotation that is 
    calculated is over the z axis. This will leave the vector v on the
    XZ plane. Then, the rotation over the y axis. 

    Returns the angles of rotation over axis z and y required to
    get the vector v into the same orientation as axis z

    Args:
        - v (): 
    """
    # Rotation over z axis 
    gamma = np.arctan(v[1]/v[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate v to calculate next rotation
    v = Rz.T@v.reshape(-1,1)
    v = v.reshape(-1)
    # Rotation over y axis
    beta = np.arctan(v[0]/v[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return Rz @ Ry

def create_cylinder(height=1, radius=None, resolution=20):
    """
    Create an cylinder in Open3D
    """
    radius = height/20 if radius is None else radius
    mesh_frame = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius,
        height=height,
        resolution=resolution)
    return(mesh_frame)

def create_segment(a, b, radius=0.05, color=(1,1,0), resolution=20):
    """
    Creates an line(cylinder) from an pointa to point b,
    or create an line from a vector v starting from origin.
    Args:
        - a, b: End points [x,y,z]
        - radius: radius cylinder
    """
    a = np.array(a)
    b = np.array(b)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = a
    v = b-a 

    height = np.linalg.norm(v)
    if height == 0: return None
    R = calculate_zy_rotation_for_arrow(v)
    mesh = create_cylinder(height, radius)
    mesh.rotate(R, center=np.array([0, 0, 0]))
    mesh.translate((a+b)/2)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh

def create_tetra(p1, p2, p3, p4, color=(1,1,0)):
    vertices = o3d.utility.Vector3dVector([p1, p2, p3, p4])
    tetras = o3d.utility.Vector4iVector([[0, 1, 2, 3]])
    mesh = o3d.geometry.TetraMesh(vertices, tetras)
    mesh.paint_uniform_color(color)
    return mesh

def create_grid(p0, p1, p2, p3, ni1, ni2, color=(0,0,0)):
    '''
    p0, p1, p2, p3 : points defining a quadrilateral
    ni1: nb of equidistant intervals on segments p0p1 and p3p2
    ni2: nb of equidistant intervals on segments p1p2 and p0p3
    '''
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    vertices = [p0, p1, p2, p3]
    lines = [[0,1],[0,3],[1,2],[2,3]]
    for i in range(1,ni1):
        l = len(vertices)
        vertices.append((p0*(ni1-i)+p1*i)/ni1)
        vertices.append((p3*(ni1-i)+p2*i)/ni1)
        lines.append([l,l+1])
    for i in range(1,ni2):
        l = len(vertices)
        vertices.append((p1*(ni2-i)+p2*i)/ni2)
        vertices.append((p0*(ni2-i)+p3*i)/ni2)
        lines.append([l,l+1])
    vertices = o3d.utility.Vector3dVector(vertices)
    lines = o3d.utility.Vector2iVector(lines)
    mesh = o3d.geometry.LineSet(vertices, lines)
    mesh.paint_uniform_color(color)
    return mesh


def create_coord_frame(origin=[0, 0, 0],size=1):
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    mesh.translate(origin)
    return mesh


# Visu3d : custom class used to visualize 3d skeleton
class Visu3D:
    def __init__(self, bg_color=[0,0,0], zoom=1, segment_radius=1):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window() 
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray(bg_color)
        # Defining callbacks - Key codes: https://www.glfw.org/docs/latest/group__keys.html
        self.vis.register_key_callback(ord("R"), self.start_rotating)
        self.vis.register_key_callback(ord("O"), self.start_oscillating)
        self.vis.register_key_callback(ord("S"), self.stop_moving)
        self.vis.register_key_callback(262, self.turn_view_right) # Right arrow
        self.vis.register_key_callback(263, self.turn_view_left)  # Left arrow
        self.vis.register_key_callback(265, self.incr_rot_speed) # Up arrow
        self.vis.register_key_callback(264, self.decr_rot_speed) # Down arrow
        self.view_control = self.vis.get_view_control()
        self.zoom = zoom
        self.segment_radius = segment_radius
        self.move = "oscillate"
        self.angle = 0
        self.direction = 1
        self.oscillate_angle = 200
        self.geometries = []

    def set_view(self):
        if self.angle_view % 4 == 0:
            ax = 0
        elif self.angle_view <= 3:
            ax = 1
        else:
            ax = -1
        if self.angle_view == 2 or self.angle_view == 6:
            az = 0
        elif 3 <= self.angle_view <= 5:
            az = 1
        else:
            az = -1
        self.view_control.set_front(np.array([ax,0,az]))
        self.view_control.set_up(np.array([0,-1,0]))

    def init_view(self):
        self.angle_view = 0
        self.rot_speed = 2
        self.set_view()
        self.view_control.set_zoom(self.zoom)
        
    def create_grid(self, p0, p1, p2, p3, ni1, ni2, color=(1,1,1)):
        '''
        p0, p1, p2, p3 : points defining a quadrilateral
        ni1: nb of equidistant intervals on segments p0p1 and p3p2
        ni2: nb of equidistant intervals on segments p1p2 and p0p3
        '''
        grid = create_grid(p0, p1, p2, p3, ni1, ni2, color)
        self.vis.add_geometry(grid)
        self.geometries.append(grid)

    def create_camera(self):
        cam = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.02, cone_radius=0.03, cylinder_height=0.1, cone_height=0.08)
        cam.paint_uniform_color([0.2,0.7,1])
        cam.compute_vertex_normals()
        self.geometries.append(cam)

    def add_geometries(self):
        for geo in self.geometries:
            self.vis.add_geometry(geo, reset_bounding_box=False)

    def add_segment(self, p1, p2, radius=None, color=[1,1,1]):
        radius = self.segment_radius if radius is None else radius
        line = create_segment(p1, p2, radius=radius, color=color)
        if line: self.vis.add_geometry(line, reset_bounding_box=False)

    def clear(self):
        self.vis.clear_geometries()

    # Callback
    def incr_rot_speed(self, vis):
        if self.move == "rotate":
            if self.rot_speed * self.direction == -1:
                self.direction = 1
            else:
                self.rot_speed += self.direction
        else:
            self.rot_speed += 1
    # Callback
    def decr_rot_speed(self, vis):
        if self.move == "rotate":
            if self.rot_speed * self.direction == 1:
                self.direction = -1
            else:
                self.rot_speed -= self.direction
        else:
            self.rot_speed = max (1, self.rot_speed-1)
    # Callback
    def turn_view_right(self, vis):
        self.angle_view = (self.angle_view + 1) %8
        self.set_view()
        self.move = None
    # Callback
    def turn_view_left(self, vis):
        self.angle_view = (self.angle_view - 1) %8
        self.set_view()
        self.move = None
    # Callback
    def start_rotating(self, vis):
        self.move = "rotate"
    # Callback
    def start_oscillating(self, vis):
        self.move = "oscillate"
        self.angle = 0
    # Callback
    def stop_moving(self, vis):
        self.move = None

    def try_move(self):
        if self.move == "rotate":
            self.view_control.rotate(self.rot_speed * self.direction,0)
        elif self.move == "oscillate":
            self.view_control.rotate(self.rot_speed * self.direction,0)
            self.angle += self.rot_speed * self.direction
            if abs(self.angle) >= self.oscillate_angle:
                self.direction = - self.direction

    def render(self):
        self.vis.poll_events()
        self.vis.update_renderer()



# if __name__ == "__main__":

#     line = create_segment([0, 0, 0], [1, 0, 0],  color=(1,0,0))
#     line2 = create_segment([1, 0, 0], [1, 1, 0], color=(0,1,0))
#     line3 = create_segment([1, 1, 0], [0, 0, 0], radius=0.1)
#     grid = create_grid([0,0,0],[0,0,1],[0,1,1],[0,1,0], 3, 2)
#     frame =create_coord_frame()
#     print(grid)
#     # Draw everything
#     o3d.visualization.draw_geometries([line, line2, line3, grid, frame])

import cv2
import numpy as np
from collections import namedtuple
from math import ceil, sqrt, pi, floor, sin, cos, atan2, gcd
from collections import  namedtuple

# To not display: RuntimeWarning: overflow encountered in exp
# in line:  scores = 1 / (1 + np.exp(-scores))
np.seterr(over='ignore')

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32
}

class Body:
    def __init__(self, pd_score=None, pd_box=None, pd_kps=None):
        """
        Attributes:
        pd_score : detection score
        pd_box : detection box [x, y, w, h], normalized [0,1] in the squared image
        pd_kps : detection keypoints coordinates [x, y], normalized [0,1] in the squared image
        rect_x_center, rect_y_center : center coordinates of the rotated bounding rectangle, normalized [0,1] in the squared image
        rect_w, rect_h : width and height of the rotated bounding rectangle, normalized in the squared image (may be > 1)
        rotation : rotation angle of rotated bounding rectangle with y-axis in radian
        rect_x_center_a, rect_y_center_a : center coordinates of the rotated bounding rectangle, in pixels in the squared image
        rect_w, rect_h : width and height of the rotated bounding rectangle, in pixels in the squared image
        rect_points : list of the 4 points coordinates of the rotated bounding rectangle, in pixels 
            expressed in the squared image during processing,
            expressed in the original rectangular image when returned to the user
        lm_score: global landmark score
        norm_landmarks : 3D landmarks coordinates in the rotated bounding rectangle, normalized [0,1]
        landmarks : 3D landmarks coordinates in the rotated bounding rectangle, in pixel in the original rectangular image
        world_landmarks : 3D landmarks coordinates in meter with mid hips point being the origin.
            The y value of landmarks_world coordinates is negative for landmarks 
            above the mid hips (like shoulders) and negative for landmarks below (like feet)
        xyz: (optionally) 3D location in camera coordinate system of reference point (mid hips or mid shoulders)
        xyz_ref: (optionally) name of the reference point ("mid_hips" or "mid_shoulders"),
        xyz_zone: (optionally) 4 int array of zone (in the source image) on which is measured depth.
            xyz_zone[0:2] is top-left zone corner in pixels, xyz_zone[2:4] is bottom-right zone corner
        """
        self.pd_score = pd_score
        self.pd_box = pd_box
        self.pd_kps = pd_kps
    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))


SSDAnchorOptions = namedtuple('SSDAnchorOptions',[
        'num_layers',
        'min_scale',
        'max_scale',
        'input_size_height',
        'input_size_width',
        'anchor_offset_x',
        'anchor_offset_y',
        'strides',
        'aspect_ratios',
        'reduce_boxes_in_lowest_layer',
        'interpolated_scale_aspect_ratio',
        'fixed_anchor_size'])

def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (min_scale + max_scale) / 2
    else:
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)

def generate_anchors(options):
    """
    option : SSDAnchorOptions
    # https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
    """
    anchors = []
    layer_id = 0
    n_strides = len(options.strides)
    while layer_id < n_strides:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []
        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while last_same_stride_layer < n_strides and \
                options.strides[last_same_stride_layer] == options.strides[layer_id]:
            scale = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer, n_strides)
            if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios += [1.0, 2.0, 0.5]
                scales += [0.1, scale, scale]
            else:
                aspect_ratios += options.aspect_ratios
                scales += [scale] * len(options.aspect_ratios)
                if options.interpolated_scale_aspect_ratio > 0:
                    if last_same_stride_layer == n_strides -1:
                        scale_next = 1.0
                    else:
                        scale_next = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer+1, n_strides)
                    scales.append(sqrt(scale * scale_next))
                    aspect_ratios.append(options.interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1
        
        for i,r in enumerate(aspect_ratios):
            ratio_sqrts = sqrt(r)
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options.strides[layer_id]
        feature_map_height = ceil(options.input_size_height / stride)
        feature_map_width = ceil(options.input_size_width / stride)

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options.anchor_offset_x) / feature_map_width
                    y_center = (y + options.anchor_offset_y) / feature_map_height
                    # new_anchor = Anchor(x_center=x_center, y_center=y_center)
                    if options.fixed_anchor_size:
                        new_anchor = [x_center, y_center, 1.0, 1.0]
                        # new_anchor.w = 1.0
                        # new_anchor.h = 1.0
                    else:
                        new_anchor = [x_center, y_center, anchor_width[anchor_id], anchor_height[anchor_id]]
                        # new_anchor.w = anchor_width[anchor_id]
                        # new_anchor.h = anchor_height[anchor_id]
                    anchors.append(new_anchor)
        
        layer_id = last_same_stride_layer
    return np.array(anchors)

def generate_blazepose_anchors():
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
    anchor_options = SSDAnchorOptions(
                                num_layers=5, 
                                min_scale=0.1484375,
                                max_scale=0.75,
                                input_size_height=224,
                                input_size_width=224,
                                anchor_offset_x=0.5,
                                anchor_offset_y=0.5,
                                strides=[8, 16, 32, 32, 32],
                                aspect_ratios= [1.0],
                                reduce_boxes_in_lowest_layer=False,
                                interpolated_scale_aspect_ratio=1.0,
                                fixed_anchor_size=True)
    return generate_anchors(anchor_options)

def decode_bboxes(score_thresh, scores, bboxes, anchors, best_only=False):
    """
    wi, hi : NN input shape
    https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
    # Decodes the detection tensors generated by the TensorFlow Lite model, based on
    # the SSD anchors and the specification in the options, into a vector of
    # detections. Each detection describes a detected object.
    Version 0.8.3.1:
    node {
    calculator: "TensorsToDetectionsCalculator"
    input_stream: "TENSORS:detection_tensors"
    input_side_packet: "ANCHORS:anchors"
    output_stream: "DETECTIONS:unfiltered_detections"
    options: {
        [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
        num_classes: 1
        num_boxes: 896
        num_coords: 12
        box_coord_offset: 0
        keypoint_coord_offset: 4
        num_keypoints: 4
        num_values_per_keypoint: 2
        sigmoid_score: true
        score_clipping_thresh: 100.0
        reverse_output_order: true
        x_scale: 128.0
        y_scale: 128.0
        h_scale: 128.0
        w_scale: 128.0
        min_score_thresh: 0.5
        }
    }

    Version 0.8.4:
    [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
      num_classes: 1
      num_boxes: 2254
      num_coords: 12
      box_coord_offset: 0
      keypoint_coord_offset: 4
      num_keypoints: 4
      num_values_per_keypoint: 2
      sigmoid_score: true
      score_clipping_thresh: 100.0
      reverse_output_order: true
      x_scale: 224.0
      y_scale: 224.0
      h_scale: 224.0
      w_scale: 224.0
      min_score_thresh: 0.5
    }

    # Bounding box in each pose detection is currently set to the bounding box of
    # the detected face. However, 4 additional key points are available in each
    # detection, which are used to further calculate a (rotated) bounding box that
    # encloses the body region of interest. Among the 4 key points, the first two
    # are for identifying the full-body region, and the second two for upper body
    # only:
    #
    # Key point 0 - mid hip center
    # Key point 1 - point that encodes size & rotation (for full body)
    # Key point 2 - mid shoulder center
    # Key point 3 - point that encodes size & rotation (for upper body)
    #

    scores: shape = [number of anchors 896]
    bboxes: shape = [ number of anchors x 12], 12 = 4 (bounding box : (cx,cy,w,h) + 8 (4 palm keypoints)
    """
    bodies = []
    scores = 1 / (1 + np.exp(-scores))
    if best_only:
        best_id = np.argmax(scores)
        if scores[best_id] < score_thresh: return bodies
        det_scores = scores[best_id:best_id+1]
        det_bboxes = bboxes[best_id:best_id+1]
        det_anchors = anchors[best_id:best_id+1]
    else:
        detection_mask = scores > score_thresh
        det_scores = scores[detection_mask]
        if det_scores.size == 0: return bodies
        det_bboxes = bboxes[detection_mask]
        det_anchors = anchors[detection_mask]
    
    scale = 224 # x_scale, y_scale, w_scale, h_scale

    # cx, cy, w, h = bboxes[i,:4]
    # cx = cx * anchor.w / wi + anchor.x_center 
    # cy = cy * anchor.h / hi + anchor.y_center
    # lx = lx * anchor.w / wi + anchor.x_center 
    # ly = ly * anchor.h / hi + anchor.y_center
    det_bboxes = det_bboxes* np.tile(det_anchors[:,2:4], 6) / scale + np.tile(det_anchors[:,0:2],6)
    # w = w * anchor.w / wi (in the prvious line, we add anchor.x_center and anchor.y_center to w and h, we need to substract them now)
    # h = h * anchor.h / hi
    det_bboxes[:,2:4] = det_bboxes[:,2:4] - det_anchors[:,0:2]
    # box = [cx - w*0.5, cy - h*0.5, w, h]
    det_bboxes[:,0:2] = det_bboxes[:,0:2] - det_bboxes[:,3:4] * 0.5

    for i in range(det_bboxes.shape[0]):
        score = det_scores[i]
        box = det_bboxes[i,0:4]
        kps = []
        for kp in range(4):
            kps.append(det_bboxes[i,4+kp*2:6+kp*2])
        bodies.append(Body(float(score), box, kps))
    return bodies


def non_max_suppression(bodies, nms_thresh):

    # cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) needs:
    # boxes = [ [x, y, w, h], ...] with x, y, w, h of type int
    # Currently, x, y, w, h are float between 0 and 1, so we arbitrarily multiply by 1000 and cast to int
    # boxes = [r.box for r in bodies]
    boxes = [ [int(x*1000) for x in r.pd_box] for r in bodies]        
    scores = [r.pd_score for r in bodies]
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh)
    return [bodies[i[0]] for i in indices]

def normalize_radians(angle):
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))

def rot_vec(vec, rotation):
    vx, vy = vec
    return [vx * cos(rotation) - vy * sin(rotation), vx * sin(rotation) + vy * cos(rotation)]

def detections_to_rect(body, kp_pair=[0,1]):
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
    # # Converts pose detection into a rectangle based on center and scale alignment
    # # points. Pose detection contains four key points: first two for full-body pose
    # # and two more for upper-body pose.
    # node {
    #   calculator: "SwitchContainer"
    #   input_side_packet: "ENABLE:upper_body_only"
    #   input_stream: "DETECTION:detection"
    #   input_stream: "IMAGE_SIZE:image_size"
    #   output_stream: "NORM_RECT:raw_roi"
    #   options {
    #     [mediapipe.SwitchContainerOptions.ext] {
    #       contained_node: {
    #         calculator: "AlignmentPointsRectsCalculator"
    #         options: {
    #           [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
    #             rotation_vector_start_keypoint_index: 0
    #             rotation_vector_end_keypoint_index: 1
    #             rotation_vector_target_angle_degrees: 90
    #           }
    #         }
    #       }
    #       contained_node: {
    #         calculator: "AlignmentPointsRectsCalculator"
    #         options: {
    #           [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
    #             rotation_vector_start_keypoint_index: 2
    #             rotation_vector_end_keypoint_index: 3
    #             rotation_vector_target_angle_degrees: 90
    #           }
    #         }
    #       }
    #     }
    #   }
    # }
    
    target_angle = pi * 0.5 # 90 = pi/2
        
    # AlignmentPointsRectsCalculator : https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
    x_center, y_center = body.pd_kps[kp_pair[0]] 
    x_scale, y_scale = body.pd_kps[kp_pair[1]] 
    # Bounding box size as double distance from center to scale point.
    box_size = sqrt((x_scale-x_center)**2 + (y_scale-y_center)**2) * 2
    body.rect_w = box_size
    body.rect_h = box_size
    body.rect_x_center = x_center
    body.rect_y_center = y_center

    rotation = target_angle - atan2(-(y_scale - y_center), x_scale - x_center)
    body.rotation = normalize_radians(rotation)
        
def rotated_rect_to_points(cx, cy, w, h, rotation):
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    points = []
    p0x = cx - a*h - b*w
    p0y = cy + b*h - a*w
    p1x = cx + a*h - b*w
    p1y = cy - b*h - a*w
    p2x = int(2*cx - p0x)
    p2y = int(2*cy - p0y)
    p3x = int(2*cx - p1x)
    p3y = int(2*cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)
    return [[p0x,p0y], [p1x,p1y], [p2x,p2y], [p3x,p3y]]

def rect_transformation(body, w, h, scale = 1.25):
    """
    w, h : image input shape
    """
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
    # # Expands pose rect with marging used during training.
    # node {
    #   calculator: "RectTransformationCalculator"
    #   input_stream: "NORM_RECT:raw_roi"
    #   input_stream: "IMAGE_SIZE:image_size"
    #   output_stream: "roi"
    #   options: {
    #     [mediapipe.RectTransformationCalculatorOptions.ext] {
    # Version 0831:
    #       scale_x: 1.5
    #       scale_y: 1.5
    # Version 084:
    #       scale_x: 1.25
    #       scale_y: 1.25
    #       square_long: true
    #     }
    #   }
    # }
    scale_x = scale
    scale_y = scale
    shift_x = 0
    shift_y = 0

    width = body.rect_w
    height = body.rect_h
    rotation = body.rotation
    if rotation == 0:
        body.rect_x_center_a = (body.rect_x_center + width * shift_x) * w
        body.rect_y_center_a = (body.rect_y_center + height * shift_y) * h
    else:
        x_shift = (w * width * shift_x * cos(rotation) - h * height * shift_y * sin(rotation)) 
        y_shift = (w * width * shift_x * sin(rotation) + h * height * shift_y * cos(rotation)) 
        body.rect_x_center_a = body.rect_x_center*w + x_shift
        body.rect_y_center_a = body.rect_y_center*h + y_shift

    # square_long: true
    long_side = max(width * w, height * h)
    body.rect_w_a = long_side * scale_x
    body.rect_h_a = long_side * scale_y
    body.rect_points = rotated_rect_to_points(body.rect_x_center_a, body.rect_y_center_a, body.rect_w_a, body.rect_h_a, body.rotation)

def warp_rect_img(rect_points, img, w, h):
        src = np.array(rect_points[1:], dtype=np.float32) # rect_points[0] is left bottom point !
        dst = np.array([(0, 0), (w, 0), (w, h)], dtype=np.float32)
        mat = cv2.getAffineTransform(src, dst)
        return cv2.warpAffine(img, mat, (w, h))

def distance(a, b):
    """
    a, b: 2 points in 3D (x,y,z)
    """
    return np.linalg.norm(a-b)

def angle(a, b, c):
    # https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
    # a, b and c : points as np.array([x, y, z]) 
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

#
def find_isp_scale_params(size, is_height=True):
    """
    Find closest valid size close to 'size' and and the corresponding parameters to setIspScale()
    This function is useful to work around a bug in depthai where ImageManip is scrambling images that have an invalid size
    is_height : boolean that indicates if the value is the height or the width of the image
    Returns: valid size, (numerator, denominator)
    """
    # We want size >= 288
    if size < 288:
        size = 288
    
    # We are looking for the list on integers that are divisible by 16 and
    # that can be written like n/d where n <= 16 and d <= 63
    if is_height:
        reference = 1080 
        other = 1920
    else:
        reference = 1920 
        other = 1080
    size_candidates = {}
    for s in range(288,reference,16):
        f = gcd(reference, s)
        n = s//f
        d = reference//f
        if n <= 16 and d <= 63 and int(round(other * n / d) % 2 == 0):
            size_candidates[s] = (n, d)
            
    # What is the candidate size closer to 'size' ?
    min_dist = -1
    for s in size_candidates:
        dist = abs(size - s)
        if min_dist == -1:
            min_dist = dist
            candidate = s
        else:
            if dist > min_dist: break
            candidate = s
            min_dist = dist
    return candidate, size_candidates[candidate]
    
#
# Filtering
#

class LandmarksSmoothingFilter: 
    '''
    Adapted from: https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmarks_smoothing_calculator.cc
    
    frequency, min_cutoff, beta, derivate_cutoff: 
                See class OneEuroFilter description.
    min_allowed_object_scale:
                If calculated object scale is less than given value smoothing will be
                disabled and landmarks will be returned as is. Default=1e-6
    disable_value_scaling:
                Disable value scaling based on object size and use `1.0` instead.
                If not disabled, value scale is calculated as inverse value of object
                size. Object size is calculated as maximum side of rectangular bounding
                box of the object in XY plane. Default=False
    '''
    def __init__(self,
                frequency=30,
                min_cutoff=1,
                beta=0,
                derivate_cutoff=1,
                min_allowed_object_scale=1e-6,
                disable_value_scaling=False
                ):
        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivate_cutoff = derivate_cutoff
        self.min_allowed_object_scale = min_allowed_object_scale
        self.disable_value_scaling = disable_value_scaling
        self.init = True

    @staticmethod
    def get_object_scale(landmarks):
        # Estimate object scale to use its inverse value as velocity scale for
        # RelativeVelocityFilter. If value will be too small (less than
        # `options_.min_allowed_object_scale`) smoothing will be disabled and
        # landmarks will be returned as is.
        # Object scale is calculated as average between bounding box width and height
        # with sides parallel to axis.
        min_xy = np.min(landmarks[:,:2], axis=0)
        max_xy = np.max(landmarks[:,:2], axis=0)
        return np.mean(max_xy - min_xy)

    def apply(self, landmarks, timestamp, object_scale=0):
        # object_scale: in practice, we use the size of the rotated rectangle region.rect_w_a=region.rect_h_a

        # Initialize filters 
        if self.init:
            self.filters = OneEuroFilter(self.frequency, self.min_cutoff, self.beta, self.derivate_cutoff)
            self.init = False

        # Get value scale as inverse value of the object scale.
        # If value is too small smoothing will be disabled and landmarks will be
        # returned as is.  
        if self.disable_value_scaling:
            value_scale = 1
        else:
            object_scale = object_scale if object_scale else self.get_object_scale(landmarks) 
            if object_scale < self.min_allowed_object_scale:
                return landmarks
            value_scale = 1 / object_scale

        return self.filters.apply(landmarks, value_scale, timestamp)

    def get_alpha(self, cutoff):
        '''
        te = 1.0 / self.frequency
        tau = 1.0 / (2 * Math.PI * cutoff)
        result = 1 / (1.0 + (tau / te))
        '''
        return 1.0 / (1.0 + (self.frequency / (2 * pi * cutoff)))

    def reset(self):
        self.init = True

class OneEuroFilter: 
    '''
    Adapted from: https://github.com/google/mediapipe/blob/master/mediapipe/util/filtering/one_euro_filter.cc
    Paper: https://cristal.univ-lille.fr/~casiez/1euro/

    frequency:  
                Frequency of incoming frames defined in seconds. Used
                only if can't be calculated from provided events (e.g.
                on the very first frame). Default=30
    min_cutoff:  
                Minimum cutoff frequency. Start by tuning this parameter while
                keeping `beta=0` to reduce jittering to the desired level. 1Hz
                (the default value) is a a good starting point.
    beta:       
                Cutoff slope. After `min_cutoff` is configured, start
                increasing `beta` value to reduce the lag introduced by the
                `min_cutoff`. Find the desired balance between jittering and lag. Default=0
    derivate_cutoff: 
                Cutoff frequency for derivate. It is set to 1Hz in the
                original algorithm, but can be turned to further smooth the
                speed (i.e. derivate) on the object. Default=1
    '''
    def __init__(self,
                frequency=30,
                min_cutoff=1,
                beta=0,
                derivate_cutoff=1,
                ):
        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivate_cutoff = derivate_cutoff
        self.x = LowPassFilter(self.get_alpha(min_cutoff))
        self.dx = LowPassFilter(self.get_alpha(derivate_cutoff))
        self.last_timestamp = 0

    def get_alpha(self, cutoff):
        '''
        te = 1.0 / self.frequency
        tau = 1.0 / (2 * Math.PI * cutoff)
        result = 1 / (1.0 + (tau / te))
        '''
        return 1.0 / (1.0 + (self.frequency / (2 * pi * cutoff)))

    def apply(self, value, value_scale, timestamp):
        '''
        Applies filter to the value.
        timestamp in s associated with the value (for instance,
        timestamp of the frame where you got value from).
        '''
        if self.last_timestamp >= timestamp:
            # Results are unpreditable in this case, so nothing to do but return same value.
            return value

        # Update the sampling frequency based on timestamps.
        if self.last_timestamp != 0 and timestamp != 0:
            self.frequency = 1 / (timestamp - self.last_timestamp)
        self.last_timestamp = timestamp

        # Estimate the current variation per second.
        if self.x.has_last_raw_value():
            dvalue = (value - self.x.last_raw_value()) * value_scale * self.frequency
        else:
            dvalue = 0
        edvalue = self.dx.apply_with_alpha(dvalue, self.get_alpha(self.derivate_cutoff))

        # Use it to update the cutoff frequency
        cutoff = self.min_cutoff + self.beta * np.abs(edvalue)

        # filter the given value.
        return self.x.apply_with_alpha(value, self.get_alpha(cutoff))
        
class LowPassFilter:
    '''
    Adapted from: https://github.com/google/mediapipe/blob/master/mediapipe/util/filtering/low_pass_filter.cc
    Note that 'value' can be a numpy array
    '''
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.initialized = False

    def apply(self, value):
        if self.initialized:
            # Regular lowpass filter.
            # result = alpha * value + (1 - alpha) * stored_value;
            result = self.alpha * value + (1 - self.alpha) * self.stored_value
        else:
            result = value
            self.initialized = True
        self.raw_value = value
        self.stored_value = result
        return result

    def apply_with_alpha(self, value, alpha):
        self.alpha = alpha
        return self.apply(value)

    def has_last_raw_value(self):
        return self.initialized

    def last_raw_value(self):
        return self.raw_value

    def last_value(self):
        return self.stored_value

    def reset(self):
        self.initialized = False






# LINE_BODY and COLORS_BODY are used when drawing the skeleton in 3D. 
rgb = {"right":(0,1,0), "left":(1,0,0), "middle":(1,1,0)}
LINES_BODY = [[9,10],[4,6],[1,3],
            [12,14],[14,16],[16,20],[20,18],[18,16],
            [12,11],[11,23],[23,24],[24,12],
            [11,13],[13,15],[15,19],[19,17],[17,15],
            [24,26],[26,28],[32,30],
            [23,25],[25,27],[29,31]]

COLORS_BODY = ["middle","right","left",
                "right","right","right","right","right",
                "middle","middle","middle","middle",
                "left","left","left","left","left",
                "right","right","right","left","left","left"]
COLORS_BODY = [rgb[x] for x in COLORS_BODY]




class BlazeposeRenderer:
    def __init__(self,
                tracker,
                show_3d=None,
                output=None):
        self.tracker = tracker
        self.show_3d = show_3d
        self.fram = None
        self.pause = False

        # Rendering flags
        self.show_rot_rect = False
        self.show_landmarks = True
        self.show_score = False
        self.show_fps = True

        self.show_xyz_zone = self.show_xyz = self.tracker.xyz

        if self.show_3d == "mixed" and not self.tracker.xyz:
            print("'mixed' 3d visualization needs the tracker to be in 'xyz' mode !")
            print("3d visualization falling back to 'world' mode.")
            self.show_3d = 'world'
        if self.show_3d == "image":
            self.vis3d = Visu3D(zoom=0.7, segment_radius=3)
            z = min(tracker.img_h, tracker.img_w)/3
            self.vis3d.create_grid([0,tracker.img_h,-z],[tracker.img_w,tracker.img_h,-z],[tracker.img_w,tracker.img_h,z],[0,tracker.img_h,z],5,2) # Floor
            self.vis3d.create_grid([0,0,z],[tracker.img_w,0,z],[tracker.img_w,tracker.img_h,z],[0,tracker.img_h,z],5,2) # Wall
            self.vis3d.init_view()
        elif self.show_3d == "world":
            self.vis3d = Visu3D(bg_color=(0.2, 0.2, 0.2), zoom=1.1, segment_radius=0.01)
            self.vis3d.create_grid([-1,1,-1],[1,1,-1],[1,1,1],[-1,1,1],2,2) # Floor
            self.vis3d.create_grid([-1,1,1],[1,1,1],[1,-1,1],[-1,-1,1],2,2) # Wall
            self.vis3d.init_view()
        elif self.show_3d == "mixed":
            self.vis3d = Visu3D(bg_color=(0.4, 0.4, 0.4), zoom=0.7, segment_radius=0.01)
            half_length = 3
            grid_depth = 5
            self.vis3d.create_grid([-half_length,1,0],[half_length,1,0],[half_length,1,grid_depth],[-half_length,1,grid_depth],2*half_length,grid_depth) # Floor
            self.vis3d.create_grid([-half_length,1,grid_depth],[half_length,1,grid_depth],[half_length,-1,grid_depth],[-half_length,-1,grid_depth],2*half_length,2) # Wall
            self.vis3d.create_camera()
            self.vis3d.init_view()

        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output,fourcc,tracker.video_fps,(tracker.img_w, tracker.img_h)) 

    def is_present(self, body, lm_id):
        return body.presence[lm_id] > self.tracker.presence_threshold

    def draw_landmarks(self, body):
        coord=[]
        
        if self.show_rot_rect:
            cv2.polylines(self.frame, [np.array(body.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
        if self.show_landmarks:                
            list_connections = LINES_BODY
            
            #print("Landmark")
            #print(np.array(body.landmarks))
            lines = [np.array([body.landmarks[point,:2] for point in line]) for line in list_connections if self.is_present(body, line[0]) and self.is_present(body, line[1])]
            cv2.polylines(self.frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)
            # for i,x_y in enumerate(body.landmarks_padded[:,:2]):
            
            #print(lines)
            for i,x_y in enumerate(body.landmarks[:self.tracker.nb_kps,:2]):
                
                if self.is_present(body, i):
                    if i > 10:
                        color = (0,255,0) if i%2==0 else (0,0,255)
                    elif i == 0:
                        color = (0,255,255)
                    elif i in [4,5,6,8,10]:
                        color = (0,255,0)
                    else:
                        color = (0,0,255)
                    #print(i,x_y)
                    coord.append(x_y[0])
                    coord.append(x_y[1])
                    cv2.circle(self.frame, (x_y[0], x_y[1]), 4, color, -11)
                else:
                    coord.append(0)
                    coord.append(0)
            #print(coord)
        if self.show_score:
            h, w = self.frame.shape[:2]
            cv2.putText(self.frame, f"Landmark score: {body.lm_score:.2f}", 
                        (20, h-60), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

        if self.show_xyz and body.xyz_ref:
            x0, y0 = body.xyz_ref_coords_pixel.astype(np.int)
            x0 -= 50
            y0 += 40
            cv2.rectangle(self.frame, (x0,y0), (x0+100, y0+85), (220,220,240), -1)
            cv2.putText(self.frame, f"X:{body.xyz[0]/10:3.0f} cm", (x0+10, y0+20), cv2.FONT_HERSHEY_PLAIN, 1, (20,180,0), 2)
            cv2.putText(self.frame, f"Y:{body.xyz[1]/10:3.0f} cm", (x0+10, y0+45), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
            cv2.putText(self.frame, f"Z:{body.xyz[2]/10:3.0f} cm", (x0+10, y0+70), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
        if self.show_xyz_zone and body.xyz_ref:
            # Show zone on which the spatial data were calculated
            cv2.rectangle(self.frame, tuple(body.xyz_zone[0:2]), tuple(body.xyz_zone[2:4]), (180,0,180), 2)
        return coord

    def draw_3d(self, body):
        self.vis3d.clear()
        self.vis3d.try_move()
        self.vis3d.add_geometries()
        if body is not None:
            points = body.landmarks if self.show_3d == "image" else body.landmarks_world
            draw_skeleton = True
            if self.show_3d == "mixed":  
                if body.xyz_ref:
                    """
                    Beware, the y value of landmarks_world coordinates is negative for landmarks 
                    above the mid hips (like shoulders) and negative for landmarks below (like feet).
                    The the y value of (x,y,z) coordinates given by depth sensor is negative in the lower part
                    of the image and positive in the upper part.
                    """
                    translation = body.xyz / 1000
                    translation[1] = -translation[1]
                    if body.xyz_ref == "mid_hips":                   
                        points = points + translation
                    elif body.xyz_ref == "mid_shoulders":
                        mid_hips_to_mid_shoulders = np.mean([
                            points[KEYPOINT_DICT['right_shoulder']],
                            points[KEYPOINT_DICT['left_shoulder']]],
                            axis=0) 
                        points = points + translation - mid_hips_to_mid_shoulders   
                else: 
                    draw_skeleton = False
            if draw_skeleton:
                lines = LINES_BODY
                colors = COLORS_BODY
                for i,a_b in enumerate(lines):
                    a, b = a_b
                    if self.is_present(body, a) and self.is_present(body, b):
                            self.vis3d.add_segment(points[a], points[b], color=colors[i])
        self.vis3d.render()
                
        
    def draw(self, frame, body):
        coord=[]
        if not self.pause:
            self.frame = frame
            cv2.rectangle(self.frame, (0,0), (self.frame.shape[1], self.frame.shape[0]), (255, 255, 255), -1)
            if body:
                coord=self.draw_landmarks(body)
            self.body = body
        elif self.frame is None:
            self.frame = frame
            self.body = None
        # else: self.frame points to previous frame
        if self.show_3d:
            self.draw_3d(self.body)
        return self.frame,coord
    
    def exit(self):
        if self.output:
            self.output.release()

    def waitKey(self, delay=1):
        if self.show_fps:
            self.tracker.fps.draw(self.frame, orig=(50,50), size=1, color=(240,180,100))
        #cv2.imshow("Blazepose", self.frame)
        if self.output:
            self.output.write(self.frame)
        key = cv2.waitKey(delay) 
        if key == 32:
            # Pause on space bar
            self.pause = not self.pause
        elif key == ord('r'):
            self.show_rot_rect = not self.show_rot_rect
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('s'):
            self.show_score = not self.show_score
        elif key == ord('f'):
            self.show_fps = not self.show_fps
        elif key == ord('x'):
            if self.tracker.xyz:
                self.show_xyz = not self.show_xyz    
        elif key == ord('z'):
            if self.tracker.xyz:
                self.show_xyz_zone = not self.show_xyz_zone 
        return key,self.frame
        
            
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")                 
parser_tracker.add_argument('-i', '--input', type=str, default="rgb", 
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default=%(default)s)")
parser_tracker.add_argument("--pd_m", type=str,
                    help="Path to an .blob file for pose detection model")
parser_tracker.add_argument("--lm_m", type=str,
                    help="Landmark model ('full' or 'lite' or 'heavy') or path to an .blob file")
parser_tracker.add_argument('-xyz', '--xyz', action="store_true", 
                    help="Get (x,y,z) coords of reference body keypoint in camera coord system (only for compatible devices)")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape before feeding pose detection model")
parser_tracker.add_argument('--no_smoothing', action="store_true", 
                    help="Disable smoothing filter")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument('--internal_frame_height', type=int, default=640,                                                                                    
                    help="Internal color camera frame height in pixels (default=%(default)i)")                    
parser_tracker.add_argument('-s', '--stats', action="store_true", 
                    help="Print some statistics at exit")
parser_tracker.add_argument('-t', '--trace', action="store_true", 
                    help="Print some debug messages")
parser_tracker.add_argument('--force_detection', action="store_true", 
                    help="Force person detection on every frame (never use landmarks from previous frame to determine ROI)")

parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-3', '--show_3d', choices=[None, "image", "world", "mixed"], default=None,
                    help="Display skeleton in 3d in a separate window. See README for description.")
parser_renderer.add_argument("-o","--output",
                    help="Path to output video file")
 

args = parser.parse_args()

import numpy as np
import cv2
from pathlib import Path
import time
import cv2

def now():
    return time.perf_counter()
    
class FPS: # To measure the number of frame per second
    def __init__(self, mean_nb_frames=10):
        self.nbf = -1
        self.fps = 0
        self.start = 0
        self.stop = 0
        self.local_start = 0
        self.mean_nb_frames = mean_nb_frames
        
    def update(self):
        if self.nbf%self.mean_nb_frames == 0:
            if self.start != 0:
                self.stop = now()
                self.fps = self.mean_nb_frames/(self.stop-self.local_start)
                self.local_start = self.stop
            else :
                self.start = self.local_start = now()    
        self.nbf+=1
    
    def get(self):
        return self.fps
    
    def get_global(self):
        if self.stop == 0: self.stop = now()
        return self.nbf/(self.stop-self.start)

    def draw(self, win, orig=(10,30), font=cv2.FONT_HERSHEY_SIMPLEX, size=2, color=(0,255,0), thickness=2):
        cv2.putText(win,f"FPS={self.get():.2f}",orig,font,size,color,thickness)


from math import sin, cos
import depthai as dai
import time, sys

SCRIPT_DIR = Path(__file__).resolve().parent
POSE_DETECTION_MODEL = str(SCRIPT_DIR / "models/pose_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/pose_landmark_full_sh4.blob")
LANDMARK_MODEL_HEAVY = str(SCRIPT_DIR / "models/pose_landmark_heavy_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/pose_landmark_lite_sh4.blob")


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()

class BlazeposeDepthai:
    """
    Blazepose body pose detector
    Arguments:
    - input_src: frame source, 
                    - "rgb" or None: OAK* internal color camera,
                    - "rgb_laconic": same as "rgb" but without sending the frames to the host,
                    - a file path of an image or a video,
                    - an integer (eg 0) for a webcam id,
    - pd_model: Blazepose detection model blob file (if None, takes the default value POSE_DETECTION_MODEL),
    - pd_score: confidence score to determine whether a detection is reliable (a float between 0 and 1).
    - lm_model: Blazepose landmark model blob file
                    - None or "full": the default blob file LANDMARK_MODEL_FULL,
                    - "lite": the default blob file LANDMARK_MODEL_LITE,
                    - "heavy": default blob file LANDMARK_MODEL_HEAVY,
                    - a path of a blob file. 
    - lm_score_thresh : confidence score to determine whether landmarks prediction is reliable (a float between 0 and 1).
    - xyz: boolean, when True get the (x, y, z) coords of the reference point (center of the hips) (if the device supports depth measures).
    - crop : boolean which indicates if square cropping is done or not
    - smoothing: boolean which indicates if smoothing filtering is applied
    - filter_window_size and filter_velocity_scale:
            The filter keeps track (on a window of specified size) of
            value changes over time, which as result gives velocity of how value
            changes over time. With higher velocity it weights new values higher.
            - higher filter_window_size adds to lag and to stability
            - lower filter_velocity_scale adds to lag and to stability

    - internal_fps : when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - resolution : sensor resolution "full" (1920x1080) or "ultra" (3840x2160),
    - internal_frame_height : when using the internal color camera, set the frame height (calling setIspScale()).
                                The width is calculated accordingly to height and depends on value of 'crop'
    - stats : boolean, when True, display some statistics when exiting.   
    - trace: boolean, when True print some debug messages   
    - force_detection:     boolean, force person detection on every frame (never use landmarks from previous frame to determine ROI)           
    """
    def __init__(self, input_src="rgb",
                pd_model=None, 
                pd_score_thresh=0.5,
                lm_model=None,
                lm_score_thresh=0.7,
                xyz=False,
                crop=False,
                smoothing= True,
                internal_fps=None,
                resolution="full",
                internal_frame_height=1080,
                stats=False,
                trace=False,
                force_detection=False
                ):
        
        self.pd_model = pd_model if pd_model else POSE_DETECTION_MODEL
        print(f"Pose detection blob file : {self.pd_model}")
        self.rect_transf_scale = 1.25
        if lm_model is None or lm_model == "full":
            self.lm_model = LANDMARK_MODEL_FULL
        elif lm_model == "lite":
            self.lm_model = LANDMARK_MODEL_LITE
        elif lm_model == "heavy":
            self.lm_model = LANDMARK_MODEL_HEAVY
        else:
            self.lm_model = lm_model
        print(f"Landmarks using blob file : {self.lm_model}")
        
        self.pd_score_thresh = pd_score_thresh
        self.lm_score_thresh = lm_score_thresh
        self.smoothing = smoothing
        self.crop = crop 
        self.internal_fps = internal_fps     
        self.stats = stats
        self.force_detection = force_detection
        self.presence_threshold = 0.5
        self.visibility_threshold = 0.5

        self.device = dai.Device()
        self.xyz = False
        
        if input_src == None or input_src == "rgb" or input_src == "rgb_laconic":
            # Note that here (in Host mode), specifying "rgb_laconic" has no effect
            # Color camera frame is systematically transferred to the host
            self.input_type = "rgb" # OAK* internal color camera
            if internal_fps is None:
                if "heavy" in str(lm_model):
                    self.internal_fps = 10
                elif "full" in str(lm_model):
                    self.internal_fps = 8
                else: # Light
                    self.internal_fps = 13
            else:
                self.internal_fps = internal_fps
            print(f"Internal camera FPS set to: {self.internal_fps}")
            if resolution == "full":
                self.resolution = (1920, 1080)
            elif resolution == "ultra":
                self.resolution = (3840, 2160)
            else:
                print(f"Error: {resolution} is not a valid resolution !")
                sys.exit()
            print("Sensor resolution:", self.resolution)

            self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps

            if xyz:
                # Check if the device supports stereo
                cameras = self.device.getConnectedCameras()
                if dai.CameraBoardSocket.LEFT in cameras and dai.CameraBoardSocket.RIGHT in cameras:
                    self.xyz = True
                else:
                    print("Warning: depth unavailable on this device, 'xyz' argument is ignored")

            if self.crop:
                self.frame_size, self.scale_nd = find_isp_scale_params(internal_frame_height)
                self.img_h = self.img_w = self.frame_size
                self.pad_w = self.pad_h = 0
                self.crop_w = (int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1])) - self.img_w) // 2

            else:
                width, self.scale_nd = find_isp_scale_params(internal_frame_height * 1920 / 1080, is_height=False)
                self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
                self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
                self.pad_h = (self.img_w - self.img_h) // 2
                self.pad_w = 0
                self.frame_size = self.img_w
                self.crop_w = 0

            print(f"Internal camera image size: {self.img_w} x {self.img_h} - crop_w:{self.crop_w} pad_h: {self.pad_h}")

        elif input_src.endswith('.jpg') or input_src.endswith('.png') :
            self.input_type= "image"
            self.img = cv2.imread(input_src)
            self.video_fps = 25
            self.img_h, self.img_w = self.img.shape[:2]
        else:
            self.input_type = "video"
            if input_src.isdigit():
                input_type = "webcam"
                input_src = int(input_src)
            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("Video FPS:", self.video_fps)

        if self.input_type != "rgb":
            print(f"Original frame size: {self.img_w}x{self.img_h}")
            if self.crop:
                self.frame_size = min(self.img_w, self.img_h) # // 16 * 16
            else:
                self.frame_size = max(self.img_w, self.img_h) #// 16 * 16
            self.crop_w = max((self.img_w - self.frame_size) // 2, 0)
            if self.crop_w: print("Cropping on width :", self.crop_w)
            self.crop_h = max((self.img_h - self.frame_size) // 2, 0)
            if self.crop_h: print("Cropping on height :", self.crop_h)

            self.pad_w = max((self.frame_size - self.img_w) // 2, 0)
            if self.pad_w: print("Padding on width :", self.pad_w)
            self.pad_h = max((self.frame_size - self.img_h) // 2, 0)
            if self.pad_h: print("Padding on height :", self.pad_h)
            
            
            print(f"Frame working size: {self.img_w}x{self.img_h}")

        self.nb_kps = 33 # Number of "viewable" keypoints

        if self.smoothing:
            
            self.filter_landmarks = LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.05,
                beta=80,
                derivate_cutoff=1
            )
            # landmarks_aux corresponds to the 2 landmarks used to compute the ROI in next frame
            self.filter_landmarks_aux = LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.01,
                beta=10,
                derivate_cutoff=1
            )
            self.filter_landmarks_world = LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.1,
                beta=40,
                derivate_cutoff=1,
                disable_value_scaling=True
            )
            if self.xyz:
                self.filter_xyz = LowPassFilter(alpha=0.25)
    
        # Create SSD anchors 
        self.anchors = generate_blazepose_anchors()
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Define and start pipeline
        self.pd_input_length = 224
        self.lm_input_length = 256
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        # Define data queues 
        if self.input_type == "rgb":
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            self.q_pre_pd_manip_cfg = self.device.getInputQueue(name="pre_pd_manip_cfg")
            if self.xyz:
                self.q_spatial_data = self.device.getOutputQueue(name="spatial_data_out", maxSize=1, blocking=False)
                self.q_spatial_config = self.device.getInputQueue("spatial_calc_config_in")

        else:
            self.q_pd_in = self.device.getInputQueue(name="pd_in")
        self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
        self.q_lm_in = self.device.getInputQueue(name="lm_in")
        self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)
            

        self.fps = FPS()

        self.nb_frames = 0
        self.nb_pd_inferences = 0
        self.nb_lm_inferences = 0
        self.nb_lm_inferences_after_landmarks_ROI = 0
        self.nb_frames_no_body = 0

        self.glob_pd_rtrip_time = 0
        self.glob_lm_rtrip_time = 0

        self.use_previous_landmarks = False

        self.cfg_pre_pd = dai.ImageManipConfig()
        self.cfg_pre_pd.setResizeThumbnail(self.pd_input_length, self.pd_input_length)

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        # pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)
        

        if self.input_type == "rgb":
            # ColorCamera
            print("Creating Color Camera...")
            cam = pipeline.createColorCamera()
            if self.resolution[0] == 1920:
                cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            else:
                cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
            cam.setInterleaved(False)
            cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
            cam.setFps(self.internal_fps)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)

            if self.crop:
                cam.setVideoSize(self.frame_size, self.frame_size)
                cam.setPreviewSize(self.frame_size, self.frame_size)
            else: 
                cam.setVideoSize(self.img_w, self.img_h)
                cam.setPreviewSize(self.img_w, self.img_h)
            
            cam_out = pipeline.createXLinkOut()
            cam_out.setStreamName("cam_out")
            cam_out.input.setQueueSize(1)
            cam_out.input.setBlocking(False)
            cam.video.link(cam_out.input)

            # Define pose detection pre processing (resize preview to (self.pd_input_length, self.pd_input_length))
            print("Creating Pose Detection pre processing image manip...")
            pre_pd_manip = pipeline.create(dai.node.ImageManip)
            pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
            pre_pd_manip.setWaitForConfigInput(True)
            pre_pd_manip.inputImage.setQueueSize(1)
            pre_pd_manip.inputImage.setBlocking(False)
            cam.preview.link(pre_pd_manip.inputImage)

            pre_pd_manip_cfg_in = pipeline.create(dai.node.XLinkIn)
            pre_pd_manip_cfg_in.setStreamName("pre_pd_manip_cfg")
            pre_pd_manip_cfg_in.out.link(pre_pd_manip.inputConfig)   

            if self.xyz:

                # For now, RGB needs fixed focus to properly align with depth.
                # This value was used during calibration
                cam.initialControl.setManualFocus(130)

                mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
                left = pipeline.createMonoCamera()
                left.setBoardSocket(dai.CameraBoardSocket.LEFT)
                left.setResolution(mono_resolution)
                left.setFps(self.internal_fps)

                right = pipeline.createMonoCamera()
                right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
                right.setResolution(mono_resolution)
                right.setFps(self.internal_fps)

                stereo = pipeline.createStereoDepth()
                stereo.setConfidenceThreshold(230)
                # LR-check is required for depth alignment
                stereo.setLeftRightCheck(True)
                stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
                stereo.setSubpixel(False)  # subpixel True -> latency

                spatial_location_calculator = pipeline.createSpatialLocationCalculator()
                spatial_location_calculator.setWaitForConfigInput(True)
                spatial_location_calculator.inputDepth.setBlocking(False)
                spatial_location_calculator.inputDepth.setQueueSize(1)

                spatial_data_out = pipeline.createXLinkOut()
                spatial_data_out.setStreamName("spatial_data_out")
                spatial_data_out.input.setQueueSize(1)
                spatial_data_out.input.setBlocking(False)

                spatial_calc_config_in = pipeline.createXLinkIn()
                spatial_calc_config_in.setStreamName("spatial_calc_config_in")

                left.out.link(stereo.left)
                right.out.link(stereo.right)    

                stereo.depth.link(spatial_location_calculator.inputDepth)

                spatial_location_calculator.out.link(spatial_data_out.input)
                spatial_calc_config_in.out.link(spatial_location_calculator.inputConfig)


        # Define pose detection model
        print("Creating Pose Detection Neural Network...")
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(str(Path(self.pd_model).resolve().absolute()))
        # Increase threads for detection
        # pd_nn.setNumInferenceThreads(2)
        # Specify that network takes latest arriving frame in non-blocking manner
        # Pose detection input                 
        if self.input_type == "rgb":
            pre_pd_manip.out.link(pd_nn.input)
        else:
            pd_in = pipeline.createXLinkIn()
            pd_in.setStreamName("pd_in")
            pd_in.out.link(pd_nn.input)

        # Pose detection output
        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)
        

        # Define landmark model
        print("Creating Landmark Neural Network...")          
        lm_nn = pipeline.createNeuralNetwork()
        lm_nn.setBlobPath(str(Path(self.lm_model).resolve().absolute()))
        lm_nn.setNumInferenceThreads(1)
        # Landmark input
        lm_in = pipeline.createXLinkIn()
        lm_in.setStreamName("lm_in")
        lm_in.out.link(lm_nn.input)
        # Landmark output
        lm_out = pipeline.createXLinkOut()
        lm_out.setStreamName("lm_out")
        lm_nn.out.link(lm_out.input)
            
        print("Pipeline created.")
        return pipeline        

    def is_present(self, body, lm_id):
        return body.presence[lm_id] > self.presence_threshold
    
    def is_visible(self, body, lm_id):
        if body.visibility[lm_id] > self.visibility_threshold and \
            0 <= body.landmarks[lm_id][0] < self.img_w and \
            0 <= body.landmarks[lm_id][1] < self.img_h :
            return True
        else:
            return False

    def query_body_xyz(self, body):
        # We want the 3d position (x,y,z) in meters of the body reference keypoint
        # in the camera coord system.
        # The reference point is either :
        # - the middle of the hips if both hips are present (presence of rght and left hips > threshold),
        # - the middle of the shoulders in case at leats one hip is not present and
        #   both shoulders are present,
        # - None otherwise
        if self.is_visible(body, KEYPOINT_DICT['right_hip']) and self.is_visible(body, KEYPOINT_DICT['left_hip']):
            body.xyz_ref = "mid_hips"
            body.xyz_ref_coords_pixel = np.mean([
                body.landmarks[KEYPOINT_DICT['right_hip']][:2],
                body.landmarks[KEYPOINT_DICT['left_hip']][:2]], 
                axis=0)
        elif self.is_visible(body, KEYPOINT_DICT['right_shoulder']) and self.is_visible(body, KEYPOINT_DICT['left_shoulder']):
            body.xyz_ref = "mid_shoulders"
            body.xyz_ref_coords_pixel = np.mean([
                body.landmarks[KEYPOINT_DICT['right_shoulder']][:2],
                body.landmarks[KEYPOINT_DICT['left_shoulder']][:2]],
                axis=0) 
        else:
            body.xyz_ref = None
            return
        # Prepare the request to SpatialLocationCalculator
        # ROI : small rectangular zone around the reference keypoint
        zone_size = max(int(body.rect_w_a / 45), 8)
        roi_center = dai.Point2f(int(body.xyz_ref_coords_pixel[0] - zone_size/2 + self.crop_w), int(body.xyz_ref_coords_pixel[1] - zone_size/2))
        roi_size = dai.Size2f(zone_size, zone_size)
        # Config
        conf_data = dai.SpatialLocationCalculatorConfigData()
        conf_data.depthThresholds.lowerThreshold = 100
        conf_data.depthThresholds.upperThreshold = 10000
        conf_data.roi = dai.Rect(roi_center, roi_size)
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.setROIs([conf_data])
        # spatial_rtrip_time = now()
        self.q_spatial_config.send(cfg)

        # Receives spatial locations
        spatial_data = self.q_spatial_data.get().getSpatialLocations()
        # self.glob_spatial_rtrip_time += now() - spatial_rtrip_time
        # self.nb_spatial_requests += 1
        sd = spatial_data[0]
        body.xyz_zone =  [
            int(sd.config.roi.topLeft().x) - self.crop_w,
            int(sd.config.roi.topLeft().y),
            int(sd.config.roi.bottomRight().x) - self.crop_w,
            int(sd.config.roi.bottomRight().y)
            ]
        body.xyz = np.array([
            sd.spatialCoordinates.x,
            sd.spatialCoordinates.y,
            sd.spatialCoordinates.z
            ])
        if self.smoothing:
            body.xyz = self.filter_xyz.apply(body.xyz)
        
    def pd_postprocess(self, inference):
        scores = np.array(inference.getLayerFp16("Identity_1"), dtype=np.float16) # 2254
        bboxes = np.array(inference.getLayerFp16("Identity"), dtype=np.float16).reshape((self.nb_anchors,12)) # 2254x12
        # Decode bboxes
        bodies = decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, best_only=True)
        if bodies:
            body = bodies[0]
        else:
            return None
        detections_to_rect(body)
        rect_transformation(body, self.frame_size, self.frame_size, self.rect_transf_scale)
        return body
   
    def lm_postprocess(self, body, inference):
        # The output names of the landmarks model are :
        # Identity_1 (1x1) : score (previously output_poseflag)
        # Identity_2 (1x128x128x1) (previously output_segmentation)
        # Identity_3 (1x64x64x39) (previously output_heatmap)
        # Identity_4 (1x117) world 3D landmarks (previously world_3d)
        # Identity (1x195) image 3D landmarks (previously ld_3d)
        body.lm_score = inference.getLayerFp16("Identity_1")[0]
        if body.lm_score > self.lm_score_thresh:  

            lm_raw = np.array(inference.getLayerFp16("Identity")).reshape(-1,5)
            # Each keypoint have 5 information:
            # - X,Y coordinates are local to the body of
            # interest and range from [0.0, 255.0].
            # - Z coordinate is measured in "image pixels" like
            # the X and Y coordinates and represents the
            # distance relative to the plane of the subject's
            # hips, which is the origin of the Z axis. Negative
            # values are between the hips and the camera;
            # positive values are behind the hips. Z coordinate
            # scale is similar with X, Y scales but has different
            # nature as obtained not via human annotation, by
            # fitting synthetic data (GHUM model) to the 2D
            # annotation.
            # - Visibility, after user-applied sigmoid denotes the
            # probability that a keypoint is located within the
            # frame and not occluded by another bigger body
            # part or another object.
            # - Presence, after user-applied sigmoid denotes the
            # probability that a keypoint is located within the
            # frame.

            # Normalize x,y,z. Scaling in z = scaling in x = 1/self.lm_input_length
            lm_raw[:,:3] /= self.lm_input_length
            # Apply sigmoid on visibility and presence (if used later)
            body.visibility = 1 / (1 + np.exp(-lm_raw[:,3]))
            body.presence = 1 / (1 + np.exp(-lm_raw[:,4]))
            
            # body.norm_landmarks contains the normalized ([0:1]) 3D coordinates of landmarks in the square rotated body bounding box
            body.norm_landmarks = lm_raw[:,:3]
            # Now calculate body.landmarks = the landmarks in the image coordinate system (in pixel) (body.landmarks)
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in body.rect_points[1:]], dtype=np.float32) # body.rect_points[0] is left bottom point and points going clockwise!
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(body.norm_landmarks[:self.nb_kps+2,:2], axis=0)
            lm_xy = np.squeeze(cv2.transform(lm_xy, mat))  
            # A segment of length 1 in the coordinates system of body bounding box takes body.rect_w_a pixels in the
            # original image. Then we arbitrarily divide by 4 for a more realistic appearance.
            lm_z = body.norm_landmarks[:self.nb_kps+2,2:3] * body.rect_w_a / 4
            lm_xyz = np.hstack((lm_xy, lm_z))

            # World landmarks are predicted in meters rather than in pixels of the image
            # and have origin in the middle of the hips rather than in the corner of the
            # pose image (cropped with given rectangle). Thus only rotation (but not scale
            # and translation) is applied to the landmarks to transform them back to
            # original  coordinates.
            body.landmarks_world = np.array(inference.getLayerFp16("Identity_4")).reshape(-1,3)[:self.nb_kps]
            sin_rot = sin(body.rotation)
            cos_rot = cos(body.rotation)
            rot_m = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
            body.landmarks_world[:,:2] = np.dot(body.landmarks_world[:,:2], rot_m)
            
            if self.smoothing:
                timestamp = now()
                object_scale = body.rect_w_a
                lm_xyz[:self.nb_kps] = self.filter_landmarks.apply(lm_xyz[:self.nb_kps], timestamp, object_scale)
                lm_xyz[self.nb_kps:] = self.filter_landmarks_aux.apply(lm_xyz[self.nb_kps:], timestamp, object_scale)
                body.landmarks_world = self.filter_landmarks_world.apply(body.landmarks_world, timestamp)

            body.landmarks = lm_xyz.astype(np.int)

            # body_from_landmarks will be used to initialize the bounding rotated rectangle in the next frame
            # The only information we need are the 2 landmarks 33 and 34
            self.body_from_landmarks = Body(pd_kps=body.landmarks[self.nb_kps:self.nb_kps+2,:2]/self.frame_size)

            # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
            if self.pad_h > 0:
                body.landmarks[:,1] -= self.pad_h
                for i in range(len(body.rect_points)):
                    body.rect_points[i][1] -= self.pad_h
            if self.pad_w > 0:
                body.landmarks[:,0] -= self.pad_w
                for i in range(len(body.rect_points)):
                    body.rect_points[i][0] -= self.pad_w
                
                
    def next_frame(self):

        self.fps.update()
           
        if self.input_type == "rgb":
            in_video = self.q_video.get()
            video_frame = in_video.getCvFrame()
            if self.pad_h:
                square_frame = cv2.copyMakeBorder(video_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)
            else:
                square_frame = video_frame
            # For debugging
            # if not self.crop:
            #     lb = self.q_lb_out.get()
            #     if lb:
            #         lb = lb.getCvFrame()
            #         cv2.imshow("letterbox", lb)
        else:
            if self.input_type == "image":
                frame = self.img.copy()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    return None, None
            # Cropping and/or padding of the video frame
            video_frame = frame[self.crop_h:self.crop_h+self.frame_size, self.crop_w:self.crop_w+self.frame_size]
            if self.pad_h or self.pad_w:
                square_frame = cv2.copyMakeBorder(video_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)
            else:
                square_frame = video_frame

        if self.force_detection or not self.use_previous_landmarks:
            if self.input_type == "rgb":
                self.q_pre_pd_manip_cfg.send(self.cfg_pre_pd)
            else:
                frame_nn = dai.ImgFrame()
                frame_nn.setTimestamp(time.monotonic())
                frame_nn.setWidth(self.pd_input_length)
                frame_nn.setHeight(self.pd_input_length)
                frame_nn.setData(to_planar(square_frame, (self.pd_input_length, self.pd_input_length)))
                pd_rtrip_time = now()
                self.q_pd_in.send(frame_nn)

            # Get pose detection
            inference = self.q_pd_out.get()
            if self.input_type != "rgb" and (self.force_detection or not self.use_previous_landmarks): 
                pd_rtrip_time = now() - pd_rtrip_time
                self.glob_pd_rtrip_time += pd_rtrip_time
            body = self.pd_postprocess(inference)
            self.nb_pd_inferences += 1
        else:
            body = self.body_from_landmarks
            detections_to_rect(body) # self.regions.pd_kps are initialized from landmarks on previous frame
            rect_transformation(body, self.frame_size, self.frame_size, self.rect_transf_scale)


        # Landmarks
        if body:
            frame_nn = warp_rect_img(body.rect_points, square_frame, self.lm_input_length, self.lm_input_length)
            frame_nn = frame_nn / 255.
            nn_data = dai.NNData()   
            nn_data.setLayer("input_1", to_planar(frame_nn, (self.lm_input_length, self.lm_input_length)))
            lm_rtrip_time = now()
            self.q_lm_in.send(nn_data)
            
            # Get landmarks
            inference = self.q_lm_out.get()
            lm_rtrip_time = now() - lm_rtrip_time
            self.glob_lm_rtrip_time += lm_rtrip_time
            self.nb_lm_inferences += 1
            self.lm_postprocess(body, inference)
            if body.lm_score < self.lm_score_thresh:
                body = None
                self.use_previous_landmarks = False
                if self.smoothing: 
                    self.filter_landmarks.reset()
                    self.filter_landmarks_aux.reset()
                    self.filter_landmarks_world.reset()
            else:
                self.use_previous_landmarks = True
                if self.xyz:
                    self.query_body_xyz(body)
            
        else:
            self.use_previous_landmarks = False
            if self.smoothing: 
                self.filter_landmarks.reset()
                self.filter_landmarks_aux.reset()
                self.filter_landmarks_world.reset()
                if self.xyz: self.filter_xyz.reset()
                
        return video_frame, body


    def exit(self):
        self.device.close()
        # Print some stats
        if self.stats:
            print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {self.fps.nbf})")
            print(f"# pose detection inferences : {self.nb_pd_inferences}")
            print(f"# landmark inferences       : {self.nb_lm_inferences}")
            if self.input_type != "rgb" and self.nb_pd_inferences != 0: print(f"Pose detection round trip   : {self.glob_pd_rtrip_time/self.nb_pd_inferences*1000:.1f} ms")
            if self.nb_lm_inferences != 0:  print(f"Landmark round trip         : {self.glob_lm_rtrip_time/self.nb_lm_inferences*1000:.1f} ms")

           


import numpy as np
from scipy import spatial
import cv2

def calculate_score(user_c,real_c):
    x1_min=1000
    x2_min=1000
    y1_min=1000
    y2_min=1000
    x1_max=0
    x2_max=0
    y1_max=0
    y2_max=0
    for i in range(len(real_c)):
        if(i%2==0):
            if(real_c[i]>0):
                x1_min=min(real_c[i],x1_min)
                x1_max=max(real_c[i],x1_max)
            if(user_c[i]>0):
                x2_min=min(user_c[i],x2_min)
                x2_max=max(user_c[i],x2_max)
        else:
            if(real_c[i]>0):
                y1_min=min(real_c[i],y1_min)
                y1_max=max(real_c[i],y1_max)
            if(user_c[i]>0):
                y2_min=min(user_c[i],y2_min)
                y2_max=max(user_c[i],y2_max)
    for i in range(len(user_c)):
        if(i%2==0):
            real_c[i]=max(0,real_c[i]-x1_min)
            try:
                real_c[i]=real_c[i]/x1_max
            except:
                real_c[i]=0
            user_c[i]=max(0,user_c[i]-x2_min)
            try:
                user_c[i]=user_c[i]/x2_max
            except:
                user_c[i]=0
            
        else:
            real_c[i]=max(0,real_c[i]-y1_min)
            try:
                real_c[i]=real_c[i]/y1_max
            except:
                real_c[i]=0
            
            user_c[i]=max(0,user_c[i]-y2_min)
            try:
                user_c[i]=user_c[i]/y2_max
            except:
                user_c[i]=0
            
    user_c = user_c/np.linalg.norm(user_c)
    real_c = real_c/np.linalg.norm(real_c)
    result = 1 - spatial.distance.cosine(user_c, real_c)
    return result
    
def get_frames():
    tracker = BlazeposeDepthai(input_src=args.input, 
                pd_model=args.pd_m,
                lm_model=args.lm_m,
                smoothing=not args.no_smoothing,   
                xyz=args.xyz,            
                crop=args.crop,
                internal_fps=args.internal_fps,
                internal_frame_height=args.internal_frame_height,
                force_detection=args.force_detection,
                stats=True,
                trace=args.trace)   

    renderer = BlazeposeRenderer(
                    tracker, 
                    show_3d=args.show_3d, 
                    output=args.output)
    SAVE_MODE=True
    if(SAVE_MODE):
        with open("output.txt", "w") as txt_file:
            while True:
                # Run blazepose on next frame
                frame, body = tracker.next_frame()
                if frame is None: break
                # Draw 2d skeleton
                frame ,coord= renderer.draw(frame, body)
                if(len(coord)>0):
                    for val in coord:
                        txt_file.write(str(val) + " ")
                    txt_file.write("\n")
                key,frame = renderer.waitKey(delay=1)
                h, w = frame.shape[:2]
                cv2.putText(frame, f"Recording Excercise (Press q to stop Recording)", 
                            (20, h-60), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
                ret, jpeg = cv2.imencode(".jpg", frame)
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                
                if key == 27 or key == ord('q'):
                    break
    else:

        with open("output.txt") as file_in:
            lines = []
            for line in file_in:
                lines.append(line)
        indx=0
        score=0
        while indx<len(lines):
            # Run blazepose on next frame
            frame, body = tracker.next_frame()
            if frame is None: break
            # Draw 2d skeleton
            frame ,coord= renderer.draw(frame, body)
            if(len(coord)>0):
                real_c=[]
                tmp=lines[indx].split(" ")
                for i in tmp:
                    try:
                        real_c.append(int(i))
                    except:
                        pass
                score+=calculate_score(coord,real_c)
                indx+=1
            
                # except:
                #     score=0
            
                key,frame = renderer.waitKey(delay=1)
                h, w = frame.shape[:2]
                tmp_score=score/indx
                if(indx<=len(lines)-3):
                    cv2.putText(frame, f"Your Current Score: {tmp_score:.2f}", 
                                (20, h-60), 
                                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
                else:
                    cv2.putText(frame, f"Your Final Score: {tmp_score:.2f}", 
                                (20, h-60), 
                                cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
                ret, jpeg = cv2.imencode(".jpg", frame)
                yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                if key == 27 or key == ord('q'):
                    break
        print(score/indx)
    renderer.exit()
    tracker.exit()

