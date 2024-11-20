import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_camera_predictor


def add_mask_overlay(frame, out_obj_ids, out_mask_logits):
    height, width = frame.shape[:2]
    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
    if mask.shape[0] == 1:
        mask = mask.squeeze(0) 
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

    red_mask = np.zeros((height, width, 3), dtype=np.uint8)
    red_mask[mask == 1] = [0, 0, 255]  
    alpha = 0.5 
    return cv2.addWeighted(frame, 1, red_mask, alpha, 0)

mouse_points = []  
def mouse_callback(event, x, y, flags, param):
    global mouse_points
    if event == cv2.EVENT_LBUTTONDOWN:  
        mouse_points.append((x, y))
        print(f"Point added: ({x}, {y})")

device = torch.device("cuda")
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

frame_count = 0
saving_enabled = False

ann_frame_idx = 0  
ann_obj_id = 1  
if_init = False

cv2.namedWindow("Camera Feed and Prediction")
cv2.setMouseCallback("Camera Feed and Prediction", mouse_callback)

try:
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Exiting...")
            break

        elif key == ord('s') and not saving_enabled:
            print("Started adding label and prediction...")
            saving_enabled = True

        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame.")
            break

        if saving_enabled:
            if not if_init:
                if len(mouse_points) > 0:  
                    points = np.array(mouse_points, dtype=np.float32)
                    labels = np.ones(len(mouse_points), dtype=np.int32)  # 所有点为前景
                    predictor.load_first_frame(frame)
                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        frame_idx=ann_frame_idx,
                        obj_id=ann_obj_id,
                        points=points,
                        labels=labels,
                    )

                    if_init = True
                    overlay = add_mask_overlay(frame, out_obj_ids, out_mask_logits)
                else:
                    cv2.putText(frame, "Click points and press 's' to start", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    overlay = np.zeros_like(frame)

            else:
                ann_frame_idx += 1
                out_obj_ids, out_mask_logits = predictor.track(frame)
                overlay = add_mask_overlay(frame, out_obj_ids, out_mask_logits)
        else:
            overlay = np.zeros_like(frame)

        combined_frame = cv2.hconcat([frame, overlay])
        cv2.imshow("Camera Feed and Prediction", combined_frame)

        cv2.resizeWindow("Camera Feed and Prediction", frame.shape[1] * 2, frame.shape[0])

finally:
    cap.release()
    cv2.destroyAllWindows()
