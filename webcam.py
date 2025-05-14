import torch
import os
import sys
import argparse
import numpy as np
import cv2 # Make sure OpenCV is installed
# from glob import glob # No longer needed for webcam
from torchvision.transforms import Normalize
import time # For FPS calculation
import traceback
# import trimesh # No longer directly needed here
# import pyrender # No longer directly needed here

# Ensure the project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from the project files
import config
import constants
from models import hmr, SMPL
from utils.imutils import crop
from utils.renderer import Renderer 

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cam_id', type=int, default=0, help='Webcam ID (usually 0 or 1)')
parser.add_argument('--checkpoint', default='data/model_checkpoint.pt', help='Path to network checkpoint')
parser.add_argument('--img_res', type=int, default=224, help='Input image resolution for the model')
parser.add_argument('--output_res', type=int, default=224, help='Resolution for the output display window')

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # --- Load Model ---
    print("Loading HMR model...")
    model = hmr(config.SMPL_MEAN_PARAMS)
    print(f"Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    checkpoint = torch.load(args.checkpoint, map_location=device)
    try:
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print("Loaded checkpoint state_dict.")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        return

    model.eval()
    model.to(device)
    print("HMR model loaded.")

    # Load SMPL model for mesh generation
    smpl_model_path = os.path.join(config.SMPL_MODEL_DIR, 'SMPL_NEUTRAL.pkl')
    print(f"Loading SMPL model from: {smpl_model_path}")
    if not os.path.exists(smpl_model_path):
          print(f"Error: SMPL model file not found at {smpl_model_path}")
          return

    try:
        smpl_neutral = SMPL(smpl_model_path, create_transl=False).to(device)
        print("SMPL model loaded.")
        if smpl_neutral.shapedirs.shape[-1] != 10:
             print(f"Error: Loaded SMPL model has {smpl_neutral.shapedirs.shape[-1]} shape components, expected 10.")
             return
        faces = smpl_neutral.faces
    except Exception as e:
        print(f"Error loading SMPL model: {e}")
        return

    # --- Initialize Webcam ---
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with ID {args.cam_id}")
        return
    print(f"Webcam {args.cam_id} opened successfully.")
    # Optional: Try setting capture resolution
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # --- Setup Renderer Instance (using output resolution) ---
    # NOTE: This renderer will use the material/lighting defined *inside* utils/renderer.py
    try:
        renderer = Renderer(focal_length=constants.FOCAL_LENGTH,
                            img_res=args.output_res, # Use output res for renderer viewport size
                            faces=faces)
        print(f"Renderer initialized with resolution {args.output_res}x{args.output_res}.")
    except Exception as e:
        print(f"Error initializing Renderer: {e}")
        print("This might be an issue with pyrender or its dependencies.")
        cap.release()
        return

    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    prev_time = time.time()

    # --- Real-time Processing Loop ---
    while True:
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
            # NOTE: This cropping/scaling is likely REQUIRED for the model
            center = np.array([orig_w / 2, orig_h / 2])
            bbox_height = min(orig_h, orig_w) * 0.8 # Simple heuristic
            scale = bbox_height / 200.0

            img_processed_np = crop(frame_rgb_float, center, scale, [args.img_res, args.img_res])
            if img_processed_np is None:
                print(f"Warning: Cropping failed for frame. Skipping.")
                cv2.imshow('Webcam SMPL Overlay', frame_display_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            img_processed_np_norm = np.transpose(img_processed_np.astype('float32'),(2,0,1))/255.0
            img_tensor = torch.from_numpy(img_processed_np_norm).float().to(device)
            norm_img = normalize_img(img_tensor).unsqueeze(0) # Add batch dimension

            # --- Model Inference ---
            with torch.no_grad():
                pred_rotmat, pred_betas, pred_camera = model(norm_img)
                pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                                            global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)
                pred_vertices = pred_output.vertices

            # --- Visualization using Renderer Class ---
            # Calculate camera translation
            pred_cam_t = torch.stack([pred_camera[:, 1],
                                        pred_camera[:, 2],
                                        2 * constants.FOCAL_LENGTH / (args.img_res * pred_camera[:, 0] + 1e-9)], dim=-1)

            # <<< REVERTED MODIFICATION START >>>
            # Prepare background image for the renderer (resized original frame to renderer's resolution)
            # --- This resize is NECESSARY for the current Renderer implementation ---
            frame_rgb_display = cv2.resize(frame_bgr[:, :, ::-1], (args.output_res, args.output_res), interpolation=cv2.INTER_LINEAR)
            frame_rgb_display_float = frame_rgb_display.astype(np.float32) / 255.0
            # <<< REVERTED MODIFICATION END >>>

            # Call the renderer instance's __call__ method
            output_img_float = renderer(pred_vertices.cpu().numpy()[0],
                                        pred_cam_t.cpu().numpy()[0],
                                        frame_rgb_display_float) # Pass resized frame as background
            output_img = (output_img_float * 255).astype(np.uint8)

            # --- Calculate FPS and Display ---
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            # Note: output_img resolution is determined by renderer init (args.output_res)
            # cv2.putText(output_img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the final image (convert RGB back to BGR for cv2)
            cv2.imshow('Webcam SMPL Overlay', output_img[:, :, ::-1])

        except Exception as e:
            print(f"\nError processing frame: {e}")
            print("------------------- Traceback -------------------")
            traceback.print_exc()
            print("-------------------------------------------------")
            # Display the original frame even if processing failed
            cv2.imshow('Webcam SMPL Overlay', frame_display_bgr)


        # --- Check for Quit Key ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # --- Cleanup ---
    print("Releasing resources.")
    cap.release()
    cv2.destroyAllWindows()
    # No need to delete renderer instance explicitly if it's managed by Python's garbage collection

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)