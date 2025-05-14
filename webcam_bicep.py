import torch
import os
import sys
import argparse
import numpy as np
import cv2 # Make sure OpenCV is installed
from torchvision.transforms import Normalize
import time # For FPS calculation
import traceback
import math # For angle calculation

# --- Imports for Socket Communication ---
import socket
import pickle
import zlib
import struct
# --------------------------------------

# --- Fallback Configuration and Constants ---
# These are defined globally and will be used if project-specific ones are not found or are incomplete.
class FallbackConfig:
    """Fallback configuration if project's config.py is not found or incomplete."""
    SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz' # Default path, adjust if necessary
    SMPL_MODEL_DIR = 'data/smpl'                   # Default path, adjust if necessary

class FallbackConstants:
    """Fallback constants if project's constants.py is not found or incomplete."""
    FOCAL_LENGTH = 5000.
    IMG_NORM_MEAN = [0.485, 0.456, 0.406]
    IMG_NORM_STD = [0.229, 0.224, 0.225]
    # --- SMPL Joint Indices (from common SMPL definitions) ---
    L_Shoulder = 16
    L_Elbow    = 18
    L_Wrist    = 20
    R_Shoulder = 17
    R_Elbow    = 19
    R_Wrist    = 21

# Initialize with fallback versions
config = FallbackConfig()
constants = FallbackConstants()

# Ensure the project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Attempt to import project-specific config and constants
try:
    import config as project_config
    config = project_config # Override fallback if successful
    print("Successfully imported project's config.py")
except ImportError:
    print("WARNING: Project's config.py not found or failed to import. Using fallback configuration.")
except Exception as e:
    print(f"WARNING: Error importing project's config.py: {e}. Using fallback configuration.")


try:
    import constants as project_constants
    constants = project_constants # Override fallback if successful
    print("Successfully imported project's constants.py")
except ImportError:
    print("WARNING: Project's constants.py not found or failed to import. Using fallback constants.")
except Exception as e:
    print(f"WARNING: Error importing project's constants.py: {e}. Using fallback constants.")


# Now import other project components
try:
    from models import hmr, SMPL
    from utils.imutils import crop
    from utils.renderer import Renderer
except ImportError as e:
    print(f"ERROR: Failed to import critical project components (models, utils). Functionality will be limited.")
    print(f"Import Error: {e}")
    # Define dummy classes if critical imports fail
    class DummyModel:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return None, None, None # Simulate model output
        def eval(self): pass
        def to(self, device): return self
        def load_state_dict(self, *args, **kwargs): pass
    class DummySMPL:
        def __init__(self, *args, **kwargs): 
            self.faces = None
            # Ensure NUM_BETAS is defined or use default for shapedirs
            num_betas_for_dummy = NUM_BETAS if 'NUM_BETAS' in globals() and isinstance(NUM_BETAS, int) else 10
            self.shapedirs = torch.zeros((1,1,num_betas_for_dummy)) 
        def __call__(self, *args, **kwargs):
            class DummyOutput:
                def __init__(self):
                    self.vertices = torch.zeros((1, 6890, 3)) # Standard SMPL vertex count
                    self.joints = torch.zeros((1, 24, 3))   # Standard 24 joints for SMPL
            return DummyOutput()
        def to(self, device): return self
    hmr = DummyModel
    SMPL = DummySMPL
    def crop(*args): return np.zeros((224,224,3)) # Dummy crop function
    class DummyRenderer:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, *args, **kwargs): return np.zeros((224,224,3)) # Dummy render output
    Renderer = DummyRenderer
    print("WARNING: Using dummy models and utilities. Pose estimation and rendering will likely not work correctly.")


# --- Socket Communication Constants ---
RELAY_HOST = 'localhost' 
RELAY_PORT = 9999        
MAGIC = b'SMPL'          
POSE_BODY_DIM = 69       
NUM_BETAS = 10 # This should ideally come from constants or be globally defined if not in constants
# -----------------------------------------------------

# --- Bicep Curl Counter Constants ---
# Adjusted for higher tolerance (easier to count)
CURL_THRESHOLD_UP = 65  # Angle considered "curled up" 
CURL_THRESHOLD_DOWN = 73 # Angle considered "extended down"

# --- Utility function for Rotation Matrix to Axis-Angle Conversion ---
def rotation_matrix_to_axis_angle(R):
    is_torch = isinstance(R, torch.Tensor)
    if is_torch:
        R_np = R.detach().cpu().numpy()
    else:
        R_np = R
    if R_np.shape != (3, 3):
        raise ValueError("Input must be a 3x3 rotation matrix")
    epsilon = 1e-6
    trace = np.trace(R_np)
    # Clip argument to arccos to prevent NaN from floating point inaccuracies
    angle_rad_val = np.clip((trace - 1) / 2.0, -1.0, 1.0)
    angle = np.arccos(angle_rad_val) 

    if np.abs(angle) < epsilon: # Rotation is close to 0
        axis = np.array([0.0, 0.0, 0.0])
    elif np.abs(angle - np.pi) < epsilon: # Rotation is close to 180 degrees
        col_norms = np.linalg.norm(R_np + np.eye(3), axis=0)
        axis_idx = np.argmax(col_norms)
        axis = (R_np + np.eye(3))[:, axis_idx]
        if np.linalg.norm(axis) < epsilon: 
            axis = np.array([1.0,0.0,0.0]) 
        else:
            axis = axis / np.linalg.norm(axis)
        axis *= np.pi # Angle is pi
    else: # General case for rotation
        axis = np.array([
            R_np[2, 1] - R_np[1, 2],
            R_np[0, 2] - R_np[2, 0],
            R_np[1, 0] - R_np[0, 1]
        ])
        sin_angle = np.sin(angle)
        if np.abs(sin_angle) < epsilon: 
            axis = np.array([0.0,0.0,0.0]) 
        else:
            axis = axis / (2 * sin_angle)
        axis *= angle # Scale unit axis by angle magnitude

    if is_torch:
        return torch.from_numpy(axis).to(R.device, dtype=R.dtype)
    else:
        return axis

# --- Function to send data over socket ---
def send_data(sock, data):
    if sock is None:
        print("Error: Socket is not connected.")
        return False
    try:
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        length  = len(payload)
        crc32   = zlib.crc32(payload) & 0xFFFFFFFF
        header  = MAGIC + struct.pack('>I I', length, crc32)
        sock.sendall(header + payload)
        return True
    except (BrokenPipeError, ConnectionResetError, EOFError, OSError) as e:
        print(f"Socket error during send: {e}")
        return False
    except pickle.PicklingError as e:
        print(f"Pickle error during send: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during send: {e}")
        traceback.print_exc()
        return False

# --- Function to calculate angle between three 3D points ---
def calculate_angle(p1, p2, p3):
    if isinstance(p1, torch.Tensor): p1 = p1.cpu().numpy()
    if isinstance(p2, torch.Tensor): p2 = p2.cpu().numpy()
    if isinstance(p3, torch.Tensor): p3 = p3.cpu().numpy()

    v1 = p1 - p2
    v2 = p3 - p2

    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)

    if mag_v1 * mag_v2 < 1e-9: 
        return 180.0 
    
    cos_angle_val = dot_product / (mag_v1 * mag_v2)
    cos_angle_val = np.clip(cos_angle_val, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle_val)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cam_id', type=int, default=0, help='Webcam ID (usually 0 or 1)')
parser.add_argument('--checkpoint', default='data/model_checkpoint.pt', help='Path to network checkpoint')
parser.add_argument('--img_res', type=int, default=224, help='Input image resolution for the model')
parser.add_argument('--output_res', type=int, default=224, help='Resolution for the output display window')
parser.add_argument('--display', action='store_true', help='Display the output overlay window')


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0) 
        print(f"Attempting to connect to relay server at {RELAY_HOST}:{RELAY_PORT}...")
        sock.connect((RELAY_HOST, RELAY_PORT))
        sock.settimeout(None) 
        print(f"Connected to relay server at {RELAY_HOST}:{RELAY_PORT}")
    except ConnectionRefusedError:
        print(f"ERROR: Connection refused. Is the relay server running and listening on port {RELAY_PORT}?")
        sock = None
    except socket.timeout:
        print(f"ERROR: Connection timed out. Is the relay server running and listening on port {RELAY_PORT}?")
        sock = None
    except Exception as e:
        print(f"ERROR: Error connecting to relay server: {e}")
        traceback.print_exc()
        sock = None

    if sock is None:
        print("Exiting due to failed socket connection.")
        return

    print("Loading HMR model...")
    mean_params_path = config.SMPL_MEAN_PARAMS if hasattr(config, 'SMPL_MEAN_PARAMS') else FallbackConfig.SMPL_MEAN_PARAMS
    if not os.path.exists(mean_params_path):
        print(f"ERROR: SMPL mean params file not found at {mean_params_path}. Check config.py or FallbackConfig.")
        if sock: sock.close(); return
    model = hmr(mean_params_path) 
    
    print(f"Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        if sock: sock.close(); return
    
    try:
        checkpoint_data = torch.load(args.checkpoint, map_location=device)
        if isinstance(checkpoint_data, dict) and 'model' in checkpoint_data:
            state_dict = checkpoint_data['model']
        else:
            state_dict = checkpoint_data 
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False) 
        print("Loaded checkpoint state_dict.")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        traceback.print_exc()
        if sock: sock.close(); return

    model.eval()
    model.to(device)
    print("HMR model loaded.")

    smpl_dir_path = config.SMPL_MODEL_DIR if hasattr(config, 'SMPL_MODEL_DIR') else FallbackConfig.SMPL_MODEL_DIR
    smpl_model_path = os.path.join(smpl_dir_path, 'SMPL_NEUTRAL.pkl') 
    print(f"Loading SMPL model from: {smpl_model_path}")
    if not os.path.exists(smpl_model_path):
        print(f"Error: SMPL model file not found at {smpl_model_path}. Check config.py or FallbackConfig.")
        if sock: sock.close(); return

    try:
        smpl_neutral = SMPL(smpl_model_path, create_transl=False).to(device)
        print("SMPL model loaded.")
        actual_betas = smpl_neutral.shapedirs.shape[-1]
        if actual_betas != NUM_BETAS:
            print(f"Warning: SMPL model has {actual_betas} shape components, but NUM_BETAS is {NUM_BETAS}. Using model's {actual_betas}.")
            # Consider NUM_BETAS = actual_betas if this discrepancy is problematic for array sizing.
    except Exception as e:
        print(f"Error loading SMPL model: {e}")
        traceback.print_exc()
        if sock: sock.close(); return
    faces = smpl_neutral.faces

    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with ID {args.cam_id}")
        if sock: sock.close(); return
    print(f"Webcam {args.cam_id} opened successfully.")

    renderer = None
    if args.display:
        try:
            focal_length_val = constants.FOCAL_LENGTH if hasattr(constants, 'FOCAL_LENGTH') else FallbackConstants.FOCAL_LENGTH
            renderer = Renderer(focal_length=focal_length_val,
                                img_res=args.output_res, 
                                faces=faces)
            print(f"Renderer initialized with resolution {args.output_res}x{args.output_res}.")
        except Exception as e:
            print(f"Error initializing Renderer: {e}")
            print("This might be an issue with pyrender or its dependencies. Display will be disabled.")
            renderer = None 
    
    norm_mean = constants.IMG_NORM_MEAN if hasattr(constants, 'IMG_NORM_MEAN') else FallbackConstants.IMG_NORM_MEAN
    norm_std = constants.IMG_NORM_STD if hasattr(constants, 'IMG_NORM_STD') else FallbackConstants.IMG_NORM_STD
    normalize_img = Normalize(mean=norm_mean, std=norm_std)
    
    print("Starting webcam stream. Press 'q' in OpenCV window (if displayed) to exit, 'r' to reset counts.") 
    print("Sending SMPL data to relay server...")

    left_curl_count = 0
    right_curl_count = 0
    left_curl_state = 'down'  
    right_curl_state = 'down' 

    while True:
        frame_start_loop = time.time()
        ret, frame_bgr = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        frame_display_bgr = frame_bgr.copy() 

        try:
            frame_rgb_float = frame_bgr[:, :, ::-1].copy().astype(np.float32)
            orig_h, orig_w = frame_rgb_float.shape[:2]

            center = np.array([orig_w / 2, orig_h / 2])
            bbox_height = min(orig_h, orig_w) * 0.8 
            scale = bbox_height / 200.0 

            img_processed_np = crop(frame_rgb_float, center, scale, [args.img_res, args.img_res])
            if img_processed_np is None:
                print(f"Warning: Cropping failed for frame. Skipping.")
                if args.display:
                    cv2.imshow('Webcam SMPL Overlay', frame_display_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): break
                    elif key == ord('r'): 
                        left_curl_count = 0
                        right_curl_count = 0
                        left_curl_state = 'down'
                        right_curl_state = 'down'
                        print("Curl counts reset.")
                continue
            
            img_processed_np_norm = np.transpose(img_processed_np.astype('float32'),(2,0,1))/255.0
            img_tensor = torch.from_numpy(img_processed_np_norm).float().to(device)
            norm_img = normalize_img(img_tensor).unsqueeze(0) 

            with torch.no_grad():
                pred_rotmat, pred_betas, pred_camera = model(norm_img)

            poses_body_send = np.zeros((1, POSE_BODY_DIM), dtype=np.float32)
            poses_root_send = np.zeros((1, 3), dtype=np.float32)
            betas_send = np.zeros((1, NUM_BETAS), dtype=np.float32) 
            trans_send = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            pred_joints_3d = None 

            if pred_rotmat is not None and pred_betas is not None and pred_camera is not None:
                if pred_rotmat.shape[0] > 0 and pred_betas.shape[0] > 0 and pred_camera.shape[0] > 0:
                    try:
                        pred_pose_aa_list = []
                        for i in range(pred_rotmat.shape[1]): 
                            rotmat_i = pred_rotmat[0, i] 
                            axis_angle_i = rotation_matrix_to_axis_angle(rotmat_i)
                            pred_pose_aa_list.append(axis_angle_i.cpu().numpy())

                        pred_pose_aa_full = np.concatenate(pred_pose_aa_list, axis=0) 
                        poses_root_extracted = pred_pose_aa_full[:3]
                        poses_body_extracted = pred_pose_aa_full[3:]
                        betas_extracted = pred_betas[0,:NUM_BETAS].cpu().numpy()
                        
                        vertical_offset = 0.8 
                        depth_z = 0.0       
                        trans_extracted = np.array([pred_camera[0, 1].item(), 
                                                    pred_camera[0, 2].item() + vertical_offset, 
                                                    depth_z], dtype=np.float32)

                        def axis_angle_to_rotation_matrix_np(axis_angle_vec):
                            theta = np.linalg.norm(axis_angle_vec)
                            if theta < 1e-9: return np.eye(3)
                            axis_norm = axis_angle_vec / theta
                            K = np.array([[0, -axis_norm[2], axis_norm[1]],
                                          [axis_norm[2], 0, -axis_norm[0]],
                                          [-axis_norm[1], axis_norm[0], 0]])
                            R_mat = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
                            return R_mat

                        global_orient_rotmat = axis_angle_to_rotation_matrix_np(poses_root_extracted)
                        rot_180_x = np.array([[1.,0.,0.], [0.,-1.,0.], [0.,0.,-1.]], dtype=np.float32)
                        rotated_global_orient_rotmat = np.dot(rot_180_x, global_orient_rotmat)
                        rotated_global_orient_aa = rotation_matrix_to_axis_angle(rotated_global_orient_rotmat)

                        poses_root_send = rotated_global_orient_aa.reshape(1, 3)
                        poses_body_send = poses_body_extracted.reshape(1, POSE_BODY_DIM)
                        betas_send = betas_extracted.reshape(1, NUM_BETAS)
                        trans_send = trans_extracted.reshape(1, 3)
                        
                        temp_smpl_output = smpl_neutral(betas=pred_betas, 
                                                        body_pose=pred_rotmat[:, 1:], 
                                                        global_orient=pred_rotmat[:, 0].unsqueeze(1), 
                                                        pose2rot=False) 
                        pred_joints_3d = temp_smpl_output.joints[0] 

                    except Exception as e:
                        print(f"ERROR: Exception during data extraction/conversion: {e}")
                        traceback.print_exc()
                else:
                    print("WARNING: Model output tensors are empty (batch size 0). Sending T-Pose.")
            else:
                print("WARNING: Model output is None. Sending T-Pose.")

            data_to_send = {
                "poses_body": poses_body_send, "poses_root": poses_root_send,
                "betas": betas_send, "trans": trans_send,
            }
            if not send_data(sock, data_to_send):
                print("ERROR: Failed to send data to relay server. Exiting.")
                break 
            
            if pred_joints_3d is not None:
                try:
                    l_shoulder_idx = constants.L_Shoulder if hasattr(constants, 'L_Shoulder') else FallbackConstants.L_Shoulder
                    l_elbow_idx    = constants.L_Elbow    if hasattr(constants, 'L_Elbow')    else FallbackConstants.L_Elbow
                    l_wrist_idx    = constants.L_Wrist    if hasattr(constants, 'L_Wrist')    else FallbackConstants.L_Wrist
                    r_shoulder_idx = constants.R_Shoulder if hasattr(constants, 'R_Shoulder') else FallbackConstants.R_Shoulder
                    r_elbow_idx    = constants.R_Elbow    if hasattr(constants, 'R_Elbow')    else FallbackConstants.R_Elbow
                    r_wrist_idx    = constants.R_Wrist    if hasattr(constants, 'R_Wrist')    else FallbackConstants.R_Wrist

                    l_shoulder = pred_joints_3d[l_shoulder_idx]
                    l_elbow    = pred_joints_3d[l_elbow_idx]
                    l_wrist    = pred_joints_3d[l_wrist_idx]
                    left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

                    if left_elbow_angle < CURL_THRESHOLD_UP and left_curl_state == 'down':
                        left_curl_state = 'up'
                    elif left_elbow_angle > CURL_THRESHOLD_DOWN and left_curl_state == 'up':
                        left_curl_state = 'down'
                        left_curl_count += 1
                        print(f"Left Curl Count: {left_curl_count}, Angle: {left_elbow_angle:.1f}")

                    r_shoulder = pred_joints_3d[r_shoulder_idx]
                    r_elbow    = pred_joints_3d[r_elbow_idx]
                    r_wrist    = pred_joints_3d[r_wrist_idx]
                    right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

                    if right_elbow_angle < CURL_THRESHOLD_UP and right_curl_state == 'down':
                        right_curl_state = 'up'
                    elif right_elbow_angle > CURL_THRESHOLD_DOWN and right_curl_state == 'up':
                        right_curl_state = 'down'
                        right_curl_count += 1
                        print(f"Right Curl Count: {right_curl_count}, Angle: {right_elbow_angle:.1f}")
                        
                except IndexError: 
                    print("Warning: Joint indices for curl counting out of bounds for the SMPL model's joint set.")
                except Exception as e:
                    print(f"Error in curl counting logic: {e}")
                    traceback.print_exc()

            output_img_display = cv2.resize(frame_display_bgr, (args.output_res, args.output_res))

            if args.display:
                if renderer is not None and pred_rotmat is not None and pred_betas is not None and pred_camera is not None:
                    try:
                        if 'temp_smpl_output' in locals() and temp_smpl_output is not None:
                             pred_vertices_render = temp_smpl_output.vertices
                        else: 
                             dummy_out_render = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:],
                                                              global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
                             pred_vertices_render = dummy_out_render.vertices
                        
                        focal_length_val = constants.FOCAL_LENGTH if hasattr(constants, 'FOCAL_LENGTH') else FallbackConstants.FOCAL_LENGTH
                        pred_cam_t_render = torch.stack([pred_camera[:,1], pred_camera[:,2],
                            2 * focal_length_val / (args.output_res * pred_camera[:,0] + 1e-9)],dim=-1) 

                        frame_rgb_display_resized = cv2.resize(frame_bgr[:,:,::-1], (args.output_res, args.output_res))
                        frame_rgb_display_float = frame_rgb_display_resized.astype(np.float32) / 255.0
                        
                        rendered_img_float = renderer(pred_vertices_render.cpu().numpy()[0],
                                                    pred_cam_t_render.cpu().numpy()[0],
                                                    frame_rgb_display_float) 
                        
                        processed_rendered_img = (rendered_img_float * 255).astype(np.uint8)
                        if processed_rendered_img.ndim == 3 and processed_rendered_img.shape[2] == 3: 
                            output_img_display = processed_rendered_img[:,:,::-1] 
                        elif processed_rendered_img.ndim == 2: 
                            output_img_display = cv2.cvtColor(processed_rendered_img, cv2.COLOR_GRAY2BGR) 
                        else: 
                            print("Warning: Renderer output has unexpected dimensions. Using resized camera frame.")

                    except Exception as e:
                        print(f"\nError during rendering: {e}")
                        traceback.print_exc()
                
                if not isinstance(output_img_display, np.ndarray) or output_img_display.size == 0:
                    print(f"Critical Error: output_img_display is invalid before putText. Type: {type(output_img_display)}. Using black fallback.")
                    output_img_display = np.zeros((args.output_res, args.output_res, 3), dtype=np.uint8)
                
                if output_img_display.dtype != np.uint8:
                    print(f"Warning: output_img_display dtype is {output_img_display.dtype}, converting to uint8.")
                    if np.issubdtype(output_img_display.dtype, np.floating): 
                        output_img_display = np.clip(output_img_display * 255, 0, 255).astype(np.uint8)
                    else: 
                        output_img_display = output_img_display.astype(np.uint8)
                
                if output_img_display.ndim == 2:
                    output_img_display = cv2.cvtColor(output_img_display, cv2.COLOR_GRAY2BGR)
                elif output_img_display.ndim == 3 and output_img_display.shape[2] == 1: 
                     output_img_display = cv2.cvtColor(output_img_display, cv2.COLOR_GRAY2BGR)

                if not output_img_display.flags['C_CONTIGUOUS']:
                    output_img_display = np.ascontiguousarray(output_img_display)

                curr_time_fps = time.time()
                fps = 1 / (curr_time_fps - frame_start_loop) if (curr_time_fps - frame_start_loop) > 0 else 0
                cv2.putText(output_img_display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(output_img_display, f"L Curls: {left_curl_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(output_img_display, f"R Curls: {right_curl_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) 
                
                if 'left_elbow_angle' in locals():
                    cv2.putText(output_img_display, f"L Elbow: {left_elbow_angle:.1f}", (output_img_display.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
                if 'right_elbow_angle' in locals():
                    cv2.putText(output_img_display, f"R Elbow: {right_elbow_angle:.1f}", (output_img_display.shape[1] - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

                cv2.imshow('Webcam SMPL Overlay', output_img_display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Exiting...")
                    break
                elif key == ord('r'): 
                    left_curl_count = 0
                    right_curl_count = 0
                    left_curl_state = 'down'
                    right_curl_state = 'down'
                    print("Curl counts reset.")
            else: 
                time.sleep(0.01) 

        except Exception as e:
            print(f"\nMAJOR ERROR PROCESSING FRAME: {e}")
            traceback.print_exc()
            if args.display: 
                cv2.imshow('Webcam SMPL Overlay', frame_display_bgr) 
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break 
                elif key == ord('r'): 
                    left_curl_count = 0
                    right_curl_count = 0
                    left_curl_state = 'down'
                    right_curl_state = 'down'
                    print("Curl counts reset.")
    
    print("Releasing resources.")
    cap.release()
    if args.display:
        cv2.destroyAllWindows()
    if sock:
        print("Closing socket connection.")
        sock.close()
    print("Webcam demo finished.")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
