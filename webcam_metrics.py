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

# --- Imports for GUI, CSV, Deque, and Threading ---
import csv
import tkinter as tk
from tkinter import ttk
import threading
from collections import deque # For metric buffering
import queue # For passing data to metric thread
# ---------------------------------

# Ensure the project root is in the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from the project files
try:
    import config
    import constants
    from models import hmr, SMPL
    from utils.imutils import crop
    from utils.renderer import Renderer
except ImportError as e:
    print(f"ERROR: Failed to import project components: {e}")
    # Fallback for basic parsing if project files are missing, script will likely fail at runtime
    class DummyConstants: 
        FOCAL_LENGTH = 5000.
        IMG_NORM_MEAN = [0.485,0.456,0.406]
        IMG_NORM_STD = [0.229,0.224,0.225]
    class DummyConfig: 
        SMPL_MODEL_DIR = 'data/smpl'
        SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
    if 'constants' not in sys.modules: 
        print("Warning: 'constants.py' not found or import failed. Using dummy values.")
        constants = DummyConstants()
    if 'config' not in sys.modules: 
        print("Warning: 'config.py' not found or import failed. Using dummy values.")
        config = DummyConfig()

# --- Socket Communication Constants ---
RELAY_HOST = 'localhost'; RELAY_PORT = 9999; MAGIC = b'SMPL'; POSE_BODY_DIM = 69; NUM_BETAS = 10

# --- Global variables for metrics and GUI ---
# For instantaneous FPS/Latency display in the OpenCV window
current_display_metrics = {"Processing FPS": 0.0, "End-to-End Latency (ms)": 0.0} 

METRIC_BUFFER_DURATION_SECONDS = 5 # Duration over which to average metrics
# Max buffer length to prevent unbounded memory use if processing is very fast or metric thread lags
MAX_BUFFER_LEN = int(60 * (METRIC_BUFFER_DURATION_SECONDS + 2)) # Buffer for ~7s at 60fps

# Buffers populated by the MAIN thread (for its own FPS/Latency)
fps_buffer = deque(maxlen=MAX_BUFFER_LEN)
latency_buffer = deque(maxlen=MAX_BUFFER_LEN)

# Buffers populated by the METRIC CALCULATION thread
pose_change_buffer = deque(maxlen=MAX_BUFFER_LEN)
translation_change_buffer = deque(maxlen=MAX_BUFFER_LEN)
joint_pos_change_buffer = deque(maxlen=MAX_BUFFER_LEN)
shape_var_buffer = deque(maxlen=MAX_BUFFER_LEN)
detection_rate_buffer = deque(maxlen=MAX_BUFFER_LEN) # Populated by metric thread based on data from main

# Stores the *previous* frame's model outputs from the MAIN thread.
# This is used to provide the metric thread with data for frame-to-frame comparisons.
main_thread_prev_model_outputs = {
    "pred_rotmat_body": None, "pred_cam_t": None, "pred_vertices": None,
    "valid_for_comparison": False # Flag to indicate if this data is valid for a diff calculation
}

CSV_FILENAME = "smpl_metrics_socket_threaded_avg.csv" # CSV filename for averaged metrics
CSV_FIELDNAMES = [ # Headers for the CSV file
    "Timestamp", "Condition", "Avg Processing FPS", "Avg End-to-End Latency (ms)",
    "Avg Pose Change (Euclidean Dist)", "Avg Translation Change (mm)",
    "Avg Joint Position Change (mm)", "Avg Shape Param Variance", "Avg Detection/Tracking Rate (%)"
]

gui_root = None; condition_var = None; app_running = True # General application control flags
# Queue for passing data packets from the main thread to the metric calculation thread
metric_data_queue = queue.Queue(maxsize=10) # maxsize helps prevent unbounded queue growth if metric thread lags
metric_thread_instance = None # Placeholder for the metric calculation thread object

# --- Utility: Rotation Matrix to Axis-Angle ---
def rotation_matrix_to_axis_angle(R):
    """
    Convert a 3x3 rotation matrix to an axis-angle representation.
    Handles both PyTorch Tensors and NumPy arrays.
    Returns NumPy array if input is NumPy, or PyTorch Tensor on the same device if input is Tensor.
    """
    is_torch = isinstance(R, torch.Tensor)
    R_np = R.detach().cpu().numpy() if is_torch else R # Convert to NumPy if it's a Tensor

    if R_np.shape != (3,3): # Basic validation
        # Return zero vector of appropriate type if input is not 3x3
        return np.zeros(3,dtype=R_np.dtype) if not is_torch else torch.zeros(3,device=R.device,dtype=R.dtype)

    epsilon=1e-6 # Small value for float comparisons
    trace=np.trace(R_np)
    acos_arg=np.clip((trace-1.0)/2.0, -1.0,1.0) # Clamp to valid range for arccos
    angle=np.arccos(acos_arg) # Calculate rotation angle

    if np.abs(angle)<epsilon: # If angle is close to zero (identity matrix)
        vec=np.array([0.,0.,0.],dtype=R_np.dtype) # Axis-angle vector is zero
    elif np.abs(angle-np.pi)<epsilon: # If angle is close to 180 degrees (pi radians)
        # For 180-degree rotations, finding the axis requires care
        xx,yy,zz=R_np[0,0],R_np[1,1],R_np[2,2]
        xy,xz,yz=R_np[0,1],R_np[0,2],R_np[1,2]
        # Determine axis based on largest diagonal component of (R+I)/2
        if xx>=yy and xx>=zz: axis=np.array([xx+1.,xy,xz])
        elif yy>=xx and yy>=zz: axis=np.array([xy,yy+1.,yz])
        else: axis=np.array([xz,yz,zz+1.])
        
        if np.linalg.norm(axis)<epsilon: # Fallback if computed axis is near zero
            axis=np.array([1.,0.,0.]) # Default to X-axis (or any arbitrary unit vector)
        
        axis=axis/np.linalg.norm(axis) # Normalize the axis
        vec=axis*angle # Scale normalized axis by angle
    else: # General case (angle not 0 or pi)
        # Axis can be found from the skew-symmetric part of R
        cand=np.array([R_np[2,1]-R_np[1,2],R_np[0,2]-R_np[2,0],R_np[1,0]-R_np[0,1]])
        if np.linalg.norm(cand)<epsilon: # If candidate axis is zero (e.g., R is symmetric, should be caught by angle checks)
            vec=np.zeros(3,dtype=R_np.dtype) # Fallback to zero vector
        else:
            axis=cand/np.linalg.norm(cand) # Normalize axis
            vec=axis*angle # Scale by angle
            
    # Return in the original type (Tensor or NumPy array)
    return torch.from_numpy(vec).to(R.device).type(R.dtype) if is_torch else vec.astype(R_np.dtype)

# --- Socket Sending ---
def send_data(sock, data):
    """Serializes and sends data over the socket with a custom header."""
    if sock is None: return False # Don't attempt to send if socket is not connected
    try:
        payload=pickle.dumps(data,protocol=pickle.HIGHEST_PROTOCOL) # Serialize data using pickle
        length=len(payload)
        crc32=zlib.crc32(payload)&0xFFFFFFFF # Calculate CRC32 checksum for data integrity
        header=MAGIC+struct.pack('>II',length,crc32) # Pack header: MAGIC (4 bytes), length (4 bytes), CRC32 (4 bytes)
        sock.sendall(header+payload) # Send header followed by payload
        return True
    except Exception as e: # Catch various potential errors (socket, pickle, struct)
        print(f"Socket/Pickle/Struct error during send_data: {e}")
        return False

# --- Metric Calculation Thread Function ---
def metrics_calculation_worker():
    """
    Worker thread that runs in the background.
    It gets processed model outputs from a queue, calculates detailed metrics,
    and populates the global metric buffers.
    """
    global app_running, metric_data_queue
    # These buffers are written to by this thread
    global pose_change_buffer, translation_change_buffer, joint_pos_change_buffer, shape_var_buffer, detection_rate_buffer

    print("Metric calculation thread started.")
    while app_running: # Loop as long as the main application is running
        try:
            # Get data packet from the main thread; timeout allows checking app_running periodically
            data_packet = metric_data_queue.get(timeout=0.1) 
            if data_packet is None: # Sentinel value (None) signals thread to terminate
                break

            # Unpack the data received from the main thread
            current_ts = data_packet["timestamp"]
            current_model_outputs = data_packet["current_outputs"] # Dict of NumPy arrays or None
            prev_model_outputs_for_comp = data_packet["prev_outputs_for_comparison"] # Dict from main_thread_prev_model_outputs
            instant_detection_rate = data_packet["detection_rate"] # Float (0.0 or 100.0)

            # Initialize metric values for this packet (default to NaN)
            instant_shape_var, instant_pose_change, instant_trans_change, instant_joint_change = np.nan, np.nan, np.nan, np.nan

            if current_model_outputs: # Proceed only if current frame's model outputs are valid
                # Extract NumPy arrays from the current_model_outputs dictionary
                pred_rotmat_curr_np = current_model_outputs["pred_rotmat_body"] # Shape (23,3,3)
                pred_cam_t_curr_np = current_model_outputs["pred_cam_t"]       # Shape (3,)
                pred_vertices_curr_np = current_model_outputs["pred_vertices"] # Shape (6890,3)
                pred_betas_curr_np = current_model_outputs["pred_betas"]       # Shape (10,)

                # Calculate shape parameter variance
                instant_shape_var = np.var(pred_betas_curr_np)

                # Calculate frame-to-frame change metrics if previous frame's data is valid for comparison
                if prev_model_outputs_for_comp and prev_model_outputs_for_comp["valid_for_comparison"]:
                    pred_rotmat_prev_np = prev_model_outputs_for_comp["pred_rotmat_body"]
                    pred_cam_t_prev_np = prev_model_outputs_for_comp["pred_cam_t"]
                    pred_vertices_prev_np = prev_model_outputs_for_comp["pred_vertices"]

                    # Pose Change
                    if pred_rotmat_prev_np is not None:
                        diff_rotmat = pred_rotmat_curr_np - pred_rotmat_prev_np
                        individual_fro_norms = np.linalg.norm(diff_rotmat, ord='fro', axis=(1,2)) # Norm for each joint's diff matrix
                        instant_pose_change = np.sum(individual_fro_norms) # Sum of norms
                    
                    # Translation Change (convert to mm)
                    if pred_cam_t_prev_np is not None:
                        # Assuming inputs (pred_cam_t_curr_np, pred_cam_t_prev_np) are in meters
                        instant_trans_change = np.linalg.norm((pred_cam_t_curr_np * 1000) - (pred_cam_t_prev_np * 1000))
                        
                    # Joint Position Change (convert to mm)
                    if pred_vertices_prev_np is not None:
                        # Assuming inputs are in meters
                        vertex_diff_mm = np.linalg.norm((pred_vertices_curr_np*1000) - (pred_vertices_prev_np*1000), axis=1)
                        instant_joint_change = np.mean(vertex_diff_mm)
            
            # Add calculated (or NaN) metrics to their respective global buffers
            pose_change_buffer.append((current_ts, instant_pose_change))
            translation_change_buffer.append((current_ts, instant_trans_change))
            joint_pos_change_buffer.append((current_ts, instant_joint_change))
            shape_var_buffer.append((current_ts, instant_shape_var))
            detection_rate_buffer.append((current_ts, instant_detection_rate)) # This was determined by main thread

            metric_data_queue.task_done() # Indicate that the fetched item has been processed

        except queue.Empty: # If queue.get times out (no data)
            continue # Loop again and check app_running flag
        except Exception as e: # Catch any other errors within the worker
            print(f"Error in metric_calculation_worker: {e}"); traceback.print_exc()
            # Optionally, put dummy/NaN values into buffers on error, or just skip
    print("Metric calculation thread finished.")

# --- GUI Functions ---
def setup_gui():
    """Initializes and runs the Tkinter GUI for user interaction."""
    global gui_root,condition_var,app_running
    gui_root=tk.Tk();gui_root.title("Metrics Recorder");gui_root.geometry("350x200")
    def on_gui_close(): global app_running; print("GUI closed by user.");app_running=False;gui_root.destroy()
    gui_root.protocol("WM_DELETE_WINDOW",on_gui_close)
    tk.Label(gui_root,text="Condition:").pack(pady=(10,0))
    conditions=["Optimal","Low Light","Partial Occlusion","Fast Motion","Static+Moving Occlusion","Custom"]
    condition_var=tk.StringVar(gui_root);condition_var.set(conditions[0]) # Default value
    ttk.OptionMenu(gui_root,condition_var,conditions[0],*conditions).pack(pady=5,padx=10,fill='x')
    tk.Button(gui_root,text="Record Avg Metrics (5s)",command=record_metrics_action,height=2).pack(pady=10,padx=10,fill='x')
    status_label_var=tk.StringVar();status_label_var.set("Press 'Record' for 5s average.")
    tk.Label(gui_root,textvariable=status_label_var,wraplength=330).pack(pady=5)
    gui_root.status_label_var=status_label_var # Save for updating messages
    try: gui_root.mainloop() # Start Tkinter event loop
    except Exception as e: print(f"Error in GUI mainloop: {e}")
    finally: print("GUI mainloop exited."); app_running=False # Ensure app stops if GUI closes

def record_metrics_action():
    """
    Called when 'Record Avg Metrics (5s)' button is pressed.
    Fetches data from metric buffers, calculates 5-second averages, and writes to CSV.
    """
    global CSV_FILENAME,CSV_FIELDNAMES,condition_var,gui_root,METRIC_BUFFER_DURATION_SECONDS
    # Buffers to read from
    global fps_buffer,latency_buffer,pose_change_buffer,translation_change_buffer,joint_pos_change_buffer,shape_var_buffer,detection_rate_buffer
    
    if not app_running: # Check if application is still running
        if gui_root and hasattr(gui_root,'status_label_var'): gui_root.status_label_var.set("Application is shutting down.")
        return

    recording_timestamp=time.time() # Timestamp for this recording event
    averaged_metrics_log={"Timestamp":time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(recording_timestamp)),"Condition":condition_var.get()}

    def get_average_from_metric_buffer(buffer_deque, nan_treatment_policy='omit'):
        """Helper to calculate average from a deque of (timestamp, value) pairs."""
        # Take a snapshot of the deque for thread-safe iteration
        # (though for simple appends/reads, direct iteration might be okay if GIL protects, but list() is safer)
        buffer_snapshot = list(buffer_deque) 
        # Filter values within the desired time window (last N seconds)
        values_in_window = [val for ts, val in buffer_snapshot if ts >= recording_timestamp - METRIC_BUFFER_DURATION_SECONDS]
        
        if not values_in_window: return np.nan # No data points in the window

        if nan_treatment_policy == 'omit': # For most metrics, ignore NaNs
            valid_numeric_values = [v for v in values_in_window if not (isinstance(v,float) and np.isnan(v))]
            if not valid_numeric_values: return np.nan # No valid numbers to average
            return np.mean(valid_numeric_values)
        elif nan_treatment_policy == 'include_zeros': # For detection rate, 0% is a valid and important value
            return np.mean(values_in_window) 
        return np.nan # Default if policy is not recognized

    # Calculate averages for all metrics
    averaged_metrics_log["Avg Processing FPS"]=get_average_from_metric_buffer(fps_buffer)
    averaged_metrics_log["Avg End-to-End Latency (ms)"]=get_average_from_metric_buffer(latency_buffer)
    averaged_metrics_log["Avg Pose Change (Euclidean Dist)"]=get_average_from_metric_buffer(pose_change_buffer)
    averaged_metrics_log["Avg Translation Change (mm)"]=get_average_from_metric_buffer(translation_change_buffer)
    averaged_metrics_log["Avg Joint Position Change (mm)"]=get_average_from_metric_buffer(joint_pos_change_buffer)
    averaged_metrics_log["Avg Shape Param Variance"]=get_average_from_metric_buffer(shape_var_buffer)
    averaged_metrics_log["Avg Detection/Tracking Rate (%)"]=get_average_from_metric_buffer(detection_rate_buffer, nan_treatment_policy='include_zeros')
    
    csv_file_exists=os.path.isfile(CSV_FILENAME)
    try:
        with open(CSV_FILENAME,'a',newline='') as csv_output_file: # Open in append mode
            csv_writer=csv.DictWriter(csv_output_file,fieldnames=CSV_FIELDNAMES)
            if not csv_file_exists: csv_writer.writeheader() # Write header if new file
            
            # Prepare row for CSV, formatting float values
            formatted_row_to_write = {}
            for csv_col_header, avg_val in averaged_metrics_log.items():
                if isinstance(avg_val, float):
                    formatted_row_to_write[csv_col_header] = f"{avg_val:.4f}" if not np.isnan(avg_val) else "NaN"
                else: # For Timestamp and Condition (strings)
                    formatted_row_to_write[csv_col_header] = avg_val
            csv_writer.writerow(formatted_row_to_write)
            
        success_message=f"Avg Metrics for '{averaged_metrics_log['Condition']}' saved to CSV."
        print(success_message);
        if gui_root and hasattr(gui_root,'status_label_var'): gui_root.status_label_var.set(success_message)
    except IOError as e_io: # Specific error for file operations
        error_message_csv=f"CSV I/O Error: {e_io}"; print(error_message_csv); traceback.print_exc()
        if gui_root and hasattr(gui_root,'status_label_var'): gui_root.status_label_var.set(error_message_csv)
    except Exception as e_gen: # Catch any other errors
        error_message_csv=f"Unexpected CSV Error: {e_gen}"; print(error_message_csv); traceback.print_exc()
        if gui_root and hasattr(gui_root,'status_label_var'): gui_root.status_label_var.set(error_message_csv)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Real-time SMPL parameter estimation from webcam, with threaded metrics calculation and optional socket sending.")
parser.add_argument('--cam_id', type=int, default=0, help='Webcam ID')
parser.add_argument('--checkpoint', default='data/model_checkpoint.pt', help='Path to HMR model checkpoint')
parser.add_argument('--img_res', type=int, default=224, help='Input image resolution for HMR model')
parser.add_argument('--output_res', type=int, default=224, help='Resolution for the OpenCV display window')
parser.add_argument('--display', action='store_true', help='Enable display of webcam feed with SMPL overlay')
parser.add_argument('--smpl_folder', default=None, help='Path to SMPL model folder (overrides config.SMPL_MODEL_DIR)')


def main(args):
    """Main function: sets up models, webcam, threads, and runs the processing loop."""
    global current_display_metrics, main_thread_prev_model_outputs, app_running, sock, metric_thread_instance
    global fps_buffer, latency_buffer # These specific buffers are populated by the main thread

    device=torch.device('cuda'if torch.cuda.is_available() else 'cpu'); print(f"Using device: {device}")
    
    # Start GUI thread
    gui_thread=threading.Thread(target=setup_gui,daemon=True); gui_thread.start(); print("GUI thread started.")
    
    # Start the metric calculation worker thread
    metric_thread_instance = threading.Thread(target=metrics_calculation_worker, daemon=True)
    metric_thread_instance.start() # Worker will wait for data on the queue

    # Initialize socket connection
    sock=None # Default to no socket
    try:
        sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM); sock.settimeout(5.0) # 5s timeout for connection
        print(f"Attempting to connect to relay server at {RELAY_HOST}:{RELAY_PORT}..."); 
        sock.connect((RELAY_HOST,RELAY_PORT))
        sock.settimeout(None); print("Successfully connected to relay server.") # Reset timeout for send operations
    except Exception as e: 
        print(f"Socket connection error: {e}. Proceeding without socket communication."); sock=None
    
    # --- Model Loading (HMR and SMPL) ---
    # (Error handling for missing files and load failures included)
    if not hasattr(config,'SMPL_MEAN_PARAMS')or not os.path.exists(config.SMPL_MEAN_PARAMS): 
        print(f"ERROR: SMPL mean_params file missing at '{getattr(config,'SMPL_MEAN_PARAMS','N/A')}'");app_running=False;return
    model=hmr(config.SMPL_MEAN_PARAMS)
    if not os.path.exists(args.checkpoint): 
        print(f"ERROR: HMR Checkpoint file missing: {args.checkpoint}");app_running=False;return
    try: 
        # Load checkpoint, trying to get 'model' key first, then the whole checkpoint if 'model' key is not present
        checkpoint_data = torch.load(args.checkpoint,map_location=device)
        model.load_state_dict(checkpoint_data.get('model', checkpoint_data),strict=False)
        print("HMR Checkpoint loaded.")
    except Exception as e:print(f"ERROR loading HMR state_dict:{e}");app_running=False;return
    model.eval().to(device)

    smpl_model_directory=args.smpl_folder if args.smpl_folder else config.SMPL_MODEL_DIR
    smpl_model_filepath=os.path.join(smpl_model_directory,'SMPL_NEUTRAL.pkl')
    if not os.path.exists(smpl_model_filepath): 
        print(f"ERROR: SMPL_NEUTRAL.pkl file missing at: {smpl_model_filepath}");app_running=False;return
    try: 
        smpl_neutral=SMPL(smpl_model_filepath,create_transl=False).to(device)
        faces=smpl_neutral.faces # Needed for renderer
        print("SMPL model loaded.")
    except Exception as e:print(f"ERROR loading SMPL model:{e}");app_running=False;return

    # --- Webcam and Renderer Setup ---
    cap=cv2.VideoCapture(args.cam_id)
    if not cap.isOpened(): print(f"ERROR: Cannot open webcam ID {args.cam_id}");app_running=False;return
    print(f"Webcam {args.cam_id} opened.")
    
    renderer=None # Initialize to None
    if args.display:
        try:
            if not hasattr(constants,'FOCAL_LENGTH'):
                print("ERROR: constants.FOCAL_LENGTH is not defined. Renderer cannot be initialized.");renderer=None
            else: 
                renderer=Renderer(focal_length=constants.FOCAL_LENGTH,img_res=args.output_res,faces=faces)
                print("Renderer initialized.")
        except Exception as e:print(f"Renderer initialization error:{e}. Display will be webcam feed only.");renderer=None
    
    # Image normalization
    if not hasattr(constants,'IMG_NORM_MEAN')or not hasattr(constants,'IMG_NORM_STD'):
        print("ERROR: IMG_NORM_MEAN or IMG_NORM_STD not defined in constants.py.");app_running=False;return
    normalize_img=Normalize(mean=constants.IMG_NORM_MEAN,std=constants.IMG_NORM_STD)
    
    print("Starting main processing loop...")
    # --- Main Real-time Processing Loop ---
    while app_running:
        frame_capture_time = time.time() # Start time for this frame's processing
        ret, frame_bgr = cap.read() # Read frame from webcam
        
        # Timestamp for all metrics derived from this frame, used for buffering
        current_frame_timestamp_for_buffer = time.time() 

        instant_detection_rate_main_thread = 0.0 # Default to failure, updated if crop succeeds
        current_model_outputs_for_metric_thread = None # Initialize as None, populated if processing succeeds

        if not ret: # If frame capture failed
            print("ERROR: Failed to capture frame from webcam.")
            # instant_detection_rate_main_thread remains 0.0
            # current_model_outputs_for_metric_thread remains None
            # Main thread's FPS/Latency for this failed attempt will still be calculated and buffered below
            if args.display: # Check if display window was closed
                try:
                    if cv2.getWindowProperty('Webcam SMPL Overlay',cv2.WND_PROP_VISIBLE)<1:app_running=False
                except cv2.error: app_running=False # Window might have been destroyed
            if cv2.waitKey(1)&0xFF==ord('q'):app_running=False # Allow quit
            if not app_running:break # Exit loop
            time.sleep(0.01) # Brief pause before trying next frame
            # Fall through to FPS/Latency calculation for this failed frame attempt and buffering
        else: # Frame captured successfully
            frame_display_bgr = frame_bgr.copy() # For display
            try: # Try block for image processing and model inference
                frame_rgb_float=frame_bgr[:,:,::-1].copy().astype(np.float32) # BGR to RGB
                orig_h,orig_w=frame_rgb_float.shape[:2]
                center_pt=np.array([orig_w/2,orig_h/2])
                bbox_h=min(orig_h,orig_w)*0.8
                img_scale=bbox_h/200.0
                
                img_processed_numpy=crop(frame_rgb_float,center_pt,img_scale,[args.img_res,args.img_res]) # Crop

                if img_processed_numpy is None: # If cropping failed
                    # instant_detection_rate_main_thread remains 0.0
                    # current_model_outputs_for_metric_thread remains None
                    # Mark that the previous data for comparison is now invalid for the *next* frame
                    main_thread_prev_model_outputs["valid_for_comparison"] = False 
                else: # Cropping successful
                    instant_detection_rate_main_thread = 100.0 # Mark detection as success
                    
                    # Prepare image for model
                    img_norm_for_model=np.transpose(img_processed_numpy.astype('float32'),(2,0,1))/255.
                    img_tensor_for_model=torch.from_numpy(img_norm_for_model).float().to(device)
                    norm_img_input_tensor=normalize_img(img_tensor_for_model).unsqueeze(0)

                    # --- HMR Model Inference ---
                    with torch.no_grad(): 
                        pred_rotmat_tensor, pred_betas_tensor, pred_camera_tensor = model(norm_img_input_tensor)
                    
                    # --- Prepare Model Outputs (as NumPy arrays) for Metric Thread and Main Thread's `prev_data` ---
                    # Body rotation matrices (23 joints, 3x3 each)
                    current_pred_rotmat_body_np = pred_rotmat_tensor[0, 1:].cpu().numpy() 
                    # Beta shape parameters (10 values)
                    current_pred_betas_np = pred_betas_tensor[0].cpu().numpy()           
                    
                    # SMPL forward pass to get vertices (needed for rendering and joint position change metric)
                    pred_output_smpl_model = smpl_neutral(betas=pred_betas_tensor, body_pose=pred_rotmat_tensor[:,1:],
                                                          global_orient=pred_rotmat_tensor[:,0].unsqueeze(1), pose2rot=False)
                    current_pred_vertices_np = pred_output_smpl_model.vertices[0].cpu().numpy() # Vertices (N_vertices, 3)
                    
                    # Calculate camera translation (consistent for metrics and rendering)
                    focal_length_const = constants.FOCAL_LENGTH if hasattr(constants,'FOCAL_LENGTH') else 5000.0 # Use fallback if needed
                    current_pred_cam_t_torch_tensor = torch.stack([
                        pred_camera_tensor[:,1], pred_camera_tensor[:,2],
                        2*focal_length_const/(args.img_res*pred_camera_tensor[:,0]+1e-9)
                    ],dim=-1)
                    current_pred_cam_t_np = current_pred_cam_t_torch_tensor[0].cpu().numpy() # Camera translation (3,)

                    # Package these NumPy arrays for the metric thread
                    current_model_outputs_for_metric_thread = {
                        "pred_rotmat_body": current_pred_rotmat_body_np,
                        "pred_betas": current_pred_betas_np,
                        "pred_vertices": current_pred_vertices_np,
                        "pred_cam_t": current_pred_cam_t_np # This is in model's coordinate system (likely meters)
                    }
                    
                    # --- Socket Sending Logic (uses original PyTorch tensors before extensive .cpu().numpy()) ---
                    if sock:
                        betas_to_send = pred_betas_tensor[0].cpu().numpy()[:NUM_BETAS].reshape(1, NUM_BETAS)
                        vert_offset_send=0.8; depth_z_send=0.0 # Parameters for server's interpretation of translation
                        trans_to_send = np.array([pred_camera_tensor[0,1].item(), 
                                                  pred_camera_tensor[0,2].item()+vert_offset_send, 
                                                  depth_z_send], dtype=np.float32).reshape(1,3)
                        
                        # Convert all 24 rotation matrices to axis-angle for sending
                        pose_aa_list_for_send = [rotation_matrix_to_axis_angle(pred_rotmat_tensor[0,i]).cpu().numpy() 
                                                 for i in range(pred_rotmat_tensor.shape[1])]
                        pose_aa_full_for_send = np.concatenate(pose_aa_list_for_send, axis=0)
                        
                        root_aa_for_send = pose_aa_full_for_send[:3]
                        body_aa_for_send = pose_aa_full_for_send[3:].reshape(1,POSE_BODY_DIM)
                        
                        # Helper for axis-angle to rotation matrix (NumPy) for the 180-deg flip
                        def aa_to_rm_numpy(aa_vec_np): 
                            angle_norm = np.linalg.norm(aa_vec_np)
                            if angle_norm < 1e-6: return np.eye(3,dtype=aa_vec_np.dtype)
                            axis_vec = aa_vec_np / angle_norm
                            K_matrix = np.array([[0,-axis_vec[2],axis_vec[1]],
                                                 [axis_vec[2],0,-axis_vec[0]],
                                                 [-axis_vec[1],axis_vec[0],0]],dtype=aa_vec_np.dtype)
                            return np.eye(3,dtype=aa_vec_np.dtype) + \
                                   np.sin(angle_norm)*K_matrix + \
                                   (1-np.cos(angle_norm))*(K_matrix@K_matrix)
                        
                        global_orient_rm_np = aa_to_rm_numpy(root_aa_for_send)
                        rot_180_x_matrix_np = np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],dtype=np.float32)
                        rotated_global_orient_rm_np = rot_180_x_matrix_np @ global_orient_rm_np
                        
                        # Convert rotated global orientation back to axis-angle for sending
                        root_pose_to_send = rotation_matrix_to_axis_angle(rotated_global_orient_rm_np).reshape(1,3)
                        
                        data_for_socket = {"poses_body":body_aa_for_send, "poses_root":root_pose_to_send,
                                           "betas":betas_to_send, "trans":trans_to_send}
                        if not send_data(sock,data_for_socket):
                            print("ERROR: Send data via socket failed. Closing socket.");
                            if sock: sock.close()
                            sock=None

                    # --- Display Rendering (if enabled) ---
                    if args.display and renderer:
                        frame_rgb_for_display=cv2.resize(frame_bgr[:,:,::-1],(args.output_res,args.output_res),interpolation=cv2.INTER_LINEAR)
                        frame_rgb_float_for_display=frame_rgb_for_display.astype(np.float32)/255.
                        # Use the NumPy versions of vertices and camera translation already prepared
                        rendered_output_img_float=renderer(current_pred_vertices_np, current_pred_cam_t_np, frame_rgb_float_for_display)
                        rendered_output_img_uint8=(rendered_output_img_float*255).astype(np.uint8)
                        frame_display_bgr=rendered_output_img_uint8[:,:,::-1] # Update BGR frame for display

            except Exception as e_proc: # Catch errors from the processing block
                print(f"ERROR during frame processing: {e_proc}"); traceback.print_exc()
                instant_detection_rate_main_thread = 0.0 # Mark detection as failed
                current_model_outputs_for_metric_thread = None # Ensure no stale/partial data is passed
                main_thread_prev_model_outputs["valid_for_comparison"] = False # Previous data is now invalid for next frame

        # --- Prepare Data Packet and Send to Metric Calculation Thread ---
        # The `main_thread_prev_model_outputs` used here is from the *previous* successful main loop iteration.
        data_packet_for_metric_thread = {
            "timestamp": current_frame_timestamp_for_buffer,
            "current_outputs": current_model_outputs_for_metric_thread, # This is None if current frame processing failed
            "prev_outputs_for_comparison": main_thread_prev_model_outputs.copy(), # Pass a copy of the *previous* state
            "detection_rate": instant_detection_rate_main_thread
        }
        try:
            metric_data_queue.put_nowait(data_packet_for_metric_thread) # Non-blocking put
        except queue.Full: # If queue is full (metric thread lagging)
            print("Warning: Metric data queue is full. Dropping current frame's data for metric thread.")

        # --- Update `main_thread_prev_model_outputs` for the *next* main loop iteration ---
        # This happens *after* sending the current packet (which used the *old* prev_data)
        if current_model_outputs_for_metric_thread: # If current frame was successfully processed
            main_thread_prev_model_outputs["pred_rotmat_body"] = current_model_outputs_for_metric_thread["pred_rotmat_body"]
            main_thread_prev_model_outputs["pred_cam_t"] = current_model_outputs_for_metric_thread["pred_cam_t"]
            main_thread_prev_model_outputs["pred_vertices"] = current_model_outputs_for_metric_thread["pred_vertices"]
            main_thread_prev_model_outputs["valid_for_comparison"] = True
        else: # Current frame failed, so for the next frame, there's no valid "previous" from this one
            main_thread_prev_model_outputs["valid_for_comparison"] = False


        # --- Main thread calculates and buffers its own FPS & Latency ---
        loop_end_time_main_thread = time.time()
        processing_time_main_thread = loop_end_time_main_thread - frame_capture_time
        instant_fps_main_thread = 1.0/processing_time_main_thread if processing_time_main_thread > 1e-6 else 0.0
        instant_latency_main_thread = processing_time_main_thread * 1000
        
        # Buffer FPS and Latency calculated by the main thread
        fps_buffer.append((current_frame_timestamp_for_buffer, instant_fps_main_thread))
        latency_buffer.append((current_frame_timestamp_for_buffer, instant_latency_main_thread))
        
        # Update metrics for on-screen display
        current_display_metrics["Processing FPS"] = instant_fps_main_thread
        current_display_metrics["End-to-End Latency (ms)"] = instant_latency_main_thread
        
        # --- Display Frame in OpenCV Window (if enabled) ---
        if args.display:
            cv2.putText(frame_display_bgr,f"FPS:{current_display_metrics['Processing FPS']:.1f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(frame_display_bgr,f"Latency:{current_display_metrics['End-to-End Latency (ms)']:.1f}ms",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            try:
                if cv2.getWindowProperty('Webcam SMPL Overlay',cv2.WND_PROP_VISIBLE)>=1: 
                    cv2.imshow('Webcam SMPL Overlay',frame_display_bgr)
                else: # Window closed by user
                    print("Display window closed (detected by getWindowProperty).")
                    app_running=False
            except cv2.error: # Window might have been destroyed
                print("Display window closed (detected by cv2.error).")
                app_running=False
            
            key_press=cv2.waitKey(1)&0xFF
            if key_press==ord('q'):app_running=False # Quit on 'q'
            elif key_press==ord('r'):record_metrics_action() # Manual record trigger
        else: # If display is off, brief sleep to yield CPU
            time.sleep(0.001) 

    # --- Cleanup actions after main loop ---
    print("Exiting main processing loop. Performing cleanup..."); 
    app_running=False # Explicitly set to ensure all threads see it

    # Signal and wait for the metric calculation thread to finish
    if metric_thread_instance and metric_thread_instance.is_alive():
        print("Signaling metric calculation thread to stop...")
        try: 
            metric_data_queue.put_nowait(None) # Send sentinel value to stop worker
        except queue.Full: 
            print("Warning: Metric queue full during shutdown signal. Worker might not get sentinel.")
        metric_thread_instance.join(timeout=2.0) # Wait for thread to terminate
        if metric_thread_instance.is_alive(): 
            print("Warning: Metric calculation thread did not join gracefully.")
    
    if cap.isOpened():cap.release() # Release webcam
    if args.display: 
        try:cv2.destroyAllWindows() # Close OpenCV windows
        except cv2.error:pass # Ignore error if windows are already gone
    
    if sock: # Close network socket
        print("Closing socket connection.");
        try: sock.shutdown(socket.SHUT_RDWR) # Politely close
        except OSError: pass # Ignore if already closed
        finally: sock.close()
        sock=None
        
    if gui_thread and gui_thread.is_alive(): # Ensure GUI thread is closed
        print("Closing GUI thread...");
        if gui_root:
            try:gui_root.destroy() # Request Tkinter to exit its mainloop
            except Exception:pass # Ignore errors if already destroyed
        gui_thread.join(timeout=2.0) # Wait for GUI thread
        if gui_thread.is_alive(): print("Warning: GUI thread did not close gracefully.")

    print("Application finished.")

if __name__ == '__main__':
    args = parser.parse_args()
    # --- Validate critical constants/config values before starting ---
    critical_configurations_ok=True
    if not hasattr(constants,'FOCAL_LENGTH'):print("ERROR: constants.FOCAL_LENGTH is not defined.");critical_configurations_ok=False
    if not hasattr(constants,'IMG_NORM_MEAN')or not hasattr(constants,'IMG_NORM_STD'):print("ERROR: constants.IMG_NORM_MEAN or constants.IMG_NORM_STD not defined.");critical_configurations_ok=False
    if not hasattr(config,'SMPL_MEAN_PARAMS'):print("ERROR: config.SMPL_MEAN_PARAMS is not defined.");critical_configurations_ok=False
    if not hasattr(config,'SMPL_MODEL_DIR')and not args.smpl_folder:print("ERROR: SMPL_MODEL_DIR not in config.py and --smpl_folder not specified.");critical_configurations_ok=False
    
    if not critical_configurations_ok:
        print("Critical configuration(s) missing. Please check constants.py and config.py. Exiting.")
        sys.exit(1) # Exit if critical configs are missing

    main(args) # Run the main application
