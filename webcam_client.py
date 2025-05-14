import torch
import os
import sys
import argparse
import numpy as np
import cv2 # Make sure OpenCV is installed
from torchvision.transforms import Normalize
import time # For FPS calculation
import traceback

# --- Imports for Socket Communication ---
import socket
import pickle
import zlib
import struct
# --------------------------------------

# Ensure the project root is in the path
# Assuming project structure where config, constants, models, utils are findable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from the project files
try:
    import config
    import constants
    from models import hmr, SMPL
    from utils.imutils import crop
    from utils.renderer import Renderer # <<< Import the Renderer class >>>
    # Need geometry utilities for pose conversion (Rotation Matrix to Axis-Angle)
    # Since webcam.py doesn't use pocolib, we'll use PyTorch/Numpy based conversion if possible
    # Implementing a simple conversion based on scipy or numpy if torch doesn't have a direct one
    # Or, if using a library like `kornia`, we could use its functions.
    # For simplicity and self-containment, we'll implement a basic conversion here.
    # A robust conversion often involves SVD or Rodrigues formula inverse.
    # Let's use a common formula based on matrix trace for axis-angle magnitude and eigenvector for axis.
    # A more direct implementation using batch_rodrigues inverse might be better if available.
    # For now, let's use a common approach.
    # Note: A more numerically stable implementation might be required for production.

except ImportError as e:
    print(f"ERROR: Failed to import project components. Make sure config, constants, models, utils are accessible.")
    print(f"Import Error: {e}")
    sys.exit(1)

# --- Socket Communication Constants (from demo2.py) ---
RELAY_HOST = 'localhost' # Address of the relay server
RELAY_PORT = 9999        # Port for communication
MAGIC = b'SMPL'          # 4-byte magic number for packet identification
POSE_BODY_DIM = 69       # Expected dimension for body pose (23 joints * 3 axis-angle)
NUM_BETAS = 10           # Standard number of shape parameters
# -----------------------------------------------------

# --- Utility function for Rotation Matrix to Axis-Angle Conversion ---
# This is a basic implementation. For robustness, consider a dedicated geometry library.
def rotation_matrix_to_axis_angle(R):
    """
    Convert a rotation matrix to axis-angle representation.
    Based on https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Vector_form
    Input:
        R: 3x3 rotation matrix (numpy array or torch tensor)
    Output:
        axis-angle vector (3,)
    """
    is_torch = isinstance(R, torch.Tensor)
    if is_torch:
        R_np = R.detach().cpu().numpy()
    else:
        R_np = R

    # Ensure R is a 3x3 matrix
    if R_np.shape != (3, 3):
        raise ValueError("Input must be a 3x3 rotation matrix")

    epsilon = 1e-6
    trace = np.trace(R_np)
    angle = np.arccos((trace - 1) / 2.0)

    if np.abs(angle) < epsilon:
        # Angle is close to 0, no rotation
        axis = np.array([0.0, 0.0, 0.0])
    elif np.abs(angle - np.pi) < epsilon:
        # Angle is close to pi (180 degrees)
        # Find eigenvector for eigenvalue 1
        v, _ = np.linalg.eig(R_np + np.eye(3))
        axis = v[:, np.isclose(_, 1.0)][:, 0]
        axis /= np.linalg.norm(axis)
        axis *= np.pi # Scale by angle
    else:
        # General case
        axis = np.array([
            R_np[2, 1] - R_np[1, 2],
            R_np[0, 2] - R_np[2, 0],
            R_np[1, 0] - R_np[0, 1]
        ])
        axis /= (2 * np.sin(angle))
        axis *= angle # Scale by angle

    if is_torch:
        return torch.from_numpy(axis).to(R.device)
    else:
        return axis

# --- Function to send data over socket (from demo2.py) ---
def send_data(sock, data):
    """Serialize and send data with size prefix."""
    if sock is None:
        print("Error: Socket is not connected.")
        return False
    try:
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        length  = len(payload)
        crc32   = zlib.crc32(payload) & 0xFFFFFFFF
        header  = MAGIC + struct.pack('>I I', length, crc32)
        sock.sendall(header + payload)                      # Send data
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

    # --- Initialize socket client ---
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Add a timeout for connection attempt
        sock.settimeout(5.0)
        print(f"Attempting to connect to relay server at {RELAY_HOST}:{RELAY_PORT}...")
        sock.connect((RELAY_HOST, RELAY_PORT))
        # Reset timeout for sending data or set a different one if needed
        sock.settimeout(None)
        print(f"Connected to relay server at {RELAY_HOST}:{RELAY_PORT}")
    except ConnectionRefusedError:
        print(f"ERROR: Connection refused. Is the relay server running and listening on port {RELAY_PORT}?")
        sock = None # Ensure sock is None if connection failed
    except socket.timeout:
         print(f"ERROR: Connection timed out. Is the relay server running and listening on port {RELAY_PORT}?")
         sock = None
    except Exception as e:
        print(f"ERROR: Error connecting to relay server: {e}")
        traceback.print_exc()
        sock = None

    # Exit if socket connection failed
    if sock is None:
        print("Exiting due to failed socket connection.")
        return

    # --- Load Model ---
    print("Loading HMR model...")
    model = hmr(config.SMPL_MEAN_PARAMS)
    print(f"Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        if sock: sock.close()
        return
    checkpoint = torch.load(args.checkpoint, map_location=device)
    try:
        state_dict = checkpoint.get('model', checkpoint)
        # Adjust keys if the checkpoint has a prefix (e.g., 'model.')
        # Example adjustment (uncomment if needed based on your checkpoint structure):
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if k.startswith('model.'):
        #         new_state_dict[k[6:]] = v
        #     else:
        #         new_state_dict[k] = v
        # model.load_state_dict(new_state_dict, strict=False)

        model.load_state_dict(state_dict, strict=False)
        print("Loaded checkpoint state_dict.")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        if sock: sock.close()
        return

    model.eval()
    model.to(device)
    print("HMR model loaded.")

    # Load SMPL model for mesh generation (needed for local display/rendering)
    smpl_model_path = os.path.join(config.SMPL_MODEL_DIR, 'SMPL_NEUTRAL.pkl')
    print(f"Loading SMPL model from: {smpl_model_path}")
    if not os.path.exists(smpl_model_path):
          print(f"Error: SMPL model file not found at {smpl_model_path}")
          if sock: sock.close()
          return

    try:
        smpl_neutral = SMPL(smpl_model_path, create_transl=False).to(device)
        print("SMPL model loaded.")
        if smpl_neutral.shapedirs.shape[-1] != 10:
             print(f"Error: Loaded SMPL model has {smpl_neutral.shapedirs.shape[-1]} shape components, expected 10.")
             if sock: sock.close()
             return
        faces = smpl_neutral.faces
    except Exception as e:
        print(f"Error loading SMPL model: {e}")
        if sock: sock.close()
        return

    # --- Initialize Webcam ---
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with ID {args.cam_id}")
        if sock: sock.close()
        return
    print(f"Webcam {args.cam_id} opened successfully.")
    # Optional: Try setting capture resolution
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # --- Setup Renderer Instance (using output resolution) ---
    # NOTE: This renderer will use the material/lighting defined *inside* utils/renderer.py
    renderer = None # Initialize renderer to None
    if args.display:
        try:
            renderer = Renderer(focal_length=constants.FOCAL_LENGTH,
                                img_res=args.output_res, # Use output res for renderer viewport size
                                faces=faces)
            print(f"Renderer initialized with resolution {args.output_res}x{args.output_res}.")
        except Exception as e:
            print(f"Error initializing Renderer: {e}")
            print("This might be an issue with pyrender or its dependencies. Display will be disabled.")
            renderer = None # Ensure renderer is None if initialization fails
            # No need to exit, just disable display

    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    prev_time = time.time()

    print("Starting webcam stream. Press 'q' in OpenCV window (if displayed) to exit.")
    print("Sending SMPL data to relay server...")

    # --- Real-time Processing Loop ---
    while True:
        frame_start_loop = time.time()
        ret, frame_bgr = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        # Keep a copy for display if processing fails
        frame_display_bgr = frame_bgr.copy()

        try:
            # Convert BGR to RGB float for processing
            frame_rgb_float = frame_bgr[:, :, ::-1].copy().astype(np.float32)
            orig_h, orig_w = frame_rgb_float.shape[:2]

            # --- Preprocessing for Model Input (using args.img_res) ---
            # This cropping/scaling is likely REQUIRED for the model
            center = np.array([orig_w / 2, orig_h / 2])
            # Simple heuristic for bbox size - can be replaced with a detector if needed
            bbox_height = min(orig_h, orig_w) * 0.8
            scale = bbox_height / 200.0 # Scale factor relative to 200px reference

            img_processed_np = crop(frame_rgb_float, center, scale, [args.img_res, args.img_res])
            if img_processed_np is None:
                print(f"Warning: Cropping failed for frame. Skipping.")
                if args.display:
                    cv2.imshow('Webcam SMPL Overlay', frame_display_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            img_processed_np_norm = np.transpose(img_processed_np.astype('float32'),(2,0,1))/255.0
            img_tensor = torch.from_numpy(img_processed_np_norm).float().to(device)
            norm_img = normalize_img(img_tensor).unsqueeze(0) # Add batch dimension

            # --- Model Inference ---
            with torch.no_grad():
                # pred_rotmat: (B, 24, 3, 3) - Rotation matrices
                # pred_betas: (B, 10) - Shape parameters
                # pred_camera: (B, 3) - Weak perspective camera [s, tx, ty]
                pred_rotmat, pred_betas, pred_camera = model(norm_img)

            # --- *** DATA EXTRACTION AND SENDING (WITH POSE CONVERSION) *** ---
            # We need to send pose as axis-angle (72 dimensions), betas (10), and translation (3)
            # The HMR model outputs rotation matrices (24, 3, 3). We need to convert these.

            poses_body_send = np.zeros((1, POSE_BODY_DIM), dtype=np.float32) # Default T-pose
            poses_root_send = np.zeros((1, 3), dtype=np.float32)           # Default root orientation
            betas_send = np.zeros((1, NUM_BETAS), dtype=np.float32)         # Default betas
            trans_send = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)      # Default translation

            data_valid = False

            if pred_rotmat is not None and pred_betas is not None and pred_camera is not None:
                 if pred_rotmat.shape[0] > 0 and pred_betas.shape[0] > 0 and pred_camera.shape[0] > 0:
                    try:
                        # Convert rotation matrices (24, 3, 3) to axis-angle (72,)
                        # We need to process each joint's rotation matrix
                        pred_pose_aa_list = []
                        for i in range(pred_rotmat.shape[1]): # Iterate through 24 joints
                            rotmat_i = pred_rotmat[0, i].cpu().numpy() # Get 3x3 matrix for joint i, batch 0
                            axis_angle_i = rotation_matrix_to_axis_angle(rotmat_i)
                            pred_pose_aa_list.append(axis_angle_i)

                        # Concatenate all axis-angle vectors
                        pred_pose_aa_full = np.concatenate(pred_pose_aa_list, axis=0) # Should be (72,)

                        # Separate global orientation (first 3) and body pose (remaining 69)
                        poses_root_extracted = pred_pose_aa_full[:3]
                        poses_body_extracted = pred_pose_aa_full[3:]

                        # Extract betas
                        betas_extracted = pred_betas[0].cpu().numpy()[:NUM_BETAS] # Get betas for batch 0

                        # Calculate translation from weak perspective camera [s, tx, ty]
                        # The camera parameters predict a 2D translation in the image plane (tx, ty)
                        # and a scale (s) related to depth.
                        # A common approach is to set a fixed depth and use tx, ty for x, y translation.
                        # The scale 's' is often related to the z-translation (depth).
                        # A simplified translation can be derived from the camera parameters:
                        # trans_x = pred_camera[0, 1]
                        # trans_y = pred_camera[0, 2]
                        # trans_z = 2 * constants.FOCAL_LENGTH / (args.img_res * pred_camera[0, 0] + 1e-9) # Depth based on scale
                        # Note: The translation logic in demo2.py's webcam mode is slightly different (adds a vertical offset).
                        # Let's use a simple mapping from the camera parameters to x, y, z translation.
                        # The exact mapping depends on how the server interprets the translation.
                        # Based on demo2.py, the camera [s, tx, ty] seems to relate to 3D translation [tx, ty, depth].
                        # Let's map pred_camera[1] to x, pred_camera[2] to y, and derive z from scale.
                        # demo2.py uses: tx = cam_person0[1], ty = cam_person0[2], depth_z = 0.0 (fixed)
                        # Let's use a similar approach, mapping pred_camera[1] to x, pred_camera[2] to y, and a fixed z.
                        # Or, map pred_camera[1] to x, pred_camera[2] to y, and derive z from pred_camera[0] (scale).
                        # Let's try mapping pred_camera[1] to x, pred_camera[2] to y, and pred_camera[0] (scale) to z (inversely).
                        # A simple mapping: x = pred_camera[1], y = pred_camera[2], z = some_factor / pred_camera[0]
                        # Or, more like demo2.py's structure: x = pred_camera[1], y = pred_camera[2] + vertical_offset, z = fixed_depth
                        # Let's use the mapping from demo2.py's webcam mode for compatibility with remote_relay.py:
                        # x = pred_camera[0, 1], y = pred_camera[0, 2] + vertical_offset, z = fixed_depth
                        # vertical_offset and fixed_depth might need tuning. Let's use values similar to demo2.py.
                        vertical_offset = 0.8 # Example vertical offset
                        depth_z = 0.0       # Example fixed depth

                        trans_extracted = np.array([pred_camera[0, 1].item(), pred_camera[0, 2].item() + vertical_offset, depth_z], dtype=np.float32)

                        # Apply 180-degree X-axis rotation to global orientation if needed by the server
                        # This was present in demo2.py's webcam mode.
                        # We need to convert the global orientation axis-angle back to rotation matrix,
                        # apply the 180 deg X rotation, and convert back to axis-angle.
                        # This requires a Rodrigues formula implementation (axis-angle to rotation matrix).

                        # Rodrigues formula: axis-angle (3,) to rotation matrix (3,3)
                        def axis_angle_to_rotation_matrix(axis_angle):
                            """
                            Convert axis-angle to rotation matrix using Rodrigues' formula.
                            Input:
                                axis_angle: axis-angle vector (3,) - numpy array
                            Output:
                                rotation matrix (3,3)
                            """
                            theta = np.linalg.norm(axis_angle)
                            if theta < 1e-6:
                                return np.eye(3)
                            axis = axis_angle / theta
                            K = np.array([
                                [0, -axis[2], axis[1]],
                                [axis[2], 0, -axis[0]],
                                [-axis[1], axis[0], 0]
                            ])
                            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
                            return R

                        # Apply the rotation fix
                        global_orient_aa = poses_root_extracted # (3,)
                        global_orient_rotmat = axis_angle_to_rotation_matrix(global_orient_aa) # (3,3)

                        # Create 180-degree rotation matrix around X-axis
                        rot_180_x = np.array([
                            [1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 0.0, -1.0]
                        ], dtype=np.float32)

                        # Apply the rotation (pre-multiply)
                        rotated_global_orient_rotmat = np.dot(rot_180_x, global_orient_rotmat)

                        # Convert back to axis-angle
                        rotated_global_orient_aa = rotation_matrix_to_axis_angle(rotated_global_orient_rotmat) # (3,)

                        # Update send variables with rotated global orientation
                        poses_root_send = rotated_global_orient_aa.reshape(1, 3)
                        poses_body_send = poses_body_extracted.reshape(1, POSE_BODY_DIM)
                        betas_send = betas_extracted.reshape(1, NUM_BETAS)
                        trans_send = trans_extracted.reshape(1, 3) # Use the calculated translation

                        data_valid = True

                    except Exception as e:
                        print(f"ERROR: Exception during data extraction/conversion: {e}")
                        traceback.print_exc()
                        # Keep default T-pose/zeros if extraction fails
                        data_valid = False
                 else:
                    print("WARNING: Model output tensors are empty (batch size 0). Sending T-Pose.")
            else:
                print("WARNING: Model output is None. Sending T-Pose.")


            # Package data
            data_to_send = {
                "poses_body": poses_body_send,
                "poses_root": poses_root_send,
                "betas": betas_send,
                "trans": trans_send,
            }

            # Send data over socket
            if not send_data(sock, data_to_send):
                print("ERROR: Failed to send data to relay server. Exiting.")
                break # Exit loop if sending fails


            # --- Optional Display ---
            if args.display and renderer is not None:
                try:
                    # Re-run SMPL model with the *original* HMR output for local rendering
                    # This is because the renderer might expect vertices derived directly
                    # from the model's output format (rotation matrices in this case).
                    # If you want to render based on the *sent* axis-angle pose, you'd need
                    # to run SMPL again with the converted axis-angle pose here.
                    # For consistency with the original webcam.py and minimal changes,
                    # we'll use the original HMR output for rendering.
                    pred_output_render = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                                      global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
                    pred_vertices_render = pred_output_render.vertices

                    # Calculate camera translation for rendering
                    pred_cam_t_render = torch.stack([pred_camera[:, 1],
                                                     pred_camera[:, 2],
                                                     2 * constants.FOCAL_LENGTH / (args.img_res * pred_camera[:, 0] + 1e-9)], dim=-1)

                    # Prepare background image for the renderer (resized original frame)
                    frame_rgb_display = cv2.resize(frame_bgr[:, :, ::-1], (args.output_res, args.output_res), interpolation=cv2.INTER_LINEAR)
                    frame_rgb_display_float = frame_rgb_display.astype(np.float32) / 255.0

                    # Call the renderer
                    output_img_float = renderer(pred_vertices_render.cpu().numpy()[0],
                                                pred_cam_t_render.cpu().numpy()[0],
                                                frame_rgb_display_float) # Pass resized frame as background
                    output_img = (output_img_float * 255).astype(np.uint8)

                    # --- Calculate FPS and Display ---
                    curr_time = time.time()
                    fps = 1 / (curr_time - frame_start_loop) if (curr_time - frame_start_loop) > 0 else 0
                    cv2.putText(output_img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # Display the final image (convert RGB back to BGR for cv2)
                    cv2.imshow('Webcam SMPL Overlay', output_img[:, :, ::-1])

                except Exception as e:
                    print(f"\nError during rendering or display: {e}")
                    print("------------------- Traceback -------------------")
                    traceback.print_exc()
                    print("-------------------------------------------------")
                    # Display the original frame even if rendering failed
                    cv2.imshow('Webcam SMPL Overlay', frame_display_bgr)

            else:
                 # If display is off, add a small delay to avoid high CPU usage
                 time.sleep(0.01)


        except Exception as e:
            print(f"\nError processing frame: {e}")
            print("------------------- Traceback -------------------")
            traceback.print_exc()
            print("-------------------------------------------------")
            # Display the original frame even if processing failed
            if args.display:
                 cv2.imshow('Webcam SMPL Overlay', frame_display_bgr)


        # --- Check for Quit Key ---
        if args.display: # Only check for key press if a window is displayed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
        else: # If no display, still allow quitting via console (requires a different mechanism, or rely on process termination)
             # For a simple script, Ctrl+C is the usual way to stop without display.
             pass


    # --- Cleanup ---
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