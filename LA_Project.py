import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import socket
import json
import math
import threading
import time

ROBOT_IP = "10.10.244.12"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

root = tk.Tk()
root.withdraw()

clicked_points = []
world_pts = []
robot_points = []
H_matrix = None
calibration_stage = 0
current_mouse_pos = (0, 0)
real_coords = (0.0, 0.0)

robot_center = None
robot_yaw = 0.0
is_moving = False


def power_method_iteration(M, num_iter=1000, e=1e-12):
    """
    Classical power iteration for a symmetric matrix M.
    Each step multiplies by M and renormalizes, which exponentially suppresses
    every eigen-direction except the one with the largest |eigenvalue|.
    The eigenvalue is recovered as the Rayleigh quotient x^T M x.
    Returns (eigenvalue, unit eigenvector).
    """
    n = M.shape[0]
    v = np.ones(n) / np.sqrt(n)

    eigval_old = eigval = 0.0
    for _ in range(num_iter):
        Mx = np.dot(M, v)
        v = Mx / np.linalg.norm(Mx)
        eigval = np.dot(v, np.dot(M, v))
        if abs(eigval - eigval_old) < e:
            break

        eigval_old = eigval

    return eigval, v


def smallest_eigenvector(M, num_iter=1000, e=1e-12):
    """
    Smallest-eigenvalue eigenvector of a symmetric PSD matrix M, via
    shifted power method (no matrix inverse, only matrix-vector products
    Step 1: find eigval_max of M with plain power iteration
    Step 2: build B = eigval_max * I - M. Its eigenvectors are the same as M's
            but its largest eigenvalue corresponds to M's smallest one
    Step 3: run power iteration on B; its dominant eigenvector is what we want
    """
    eigval_max, _ = power_method_iteration(M, num_iter, e)

    n = M.shape[0]
    B = eigval_max * np.eye(n) - M

    _, v_min = power_method_iteration(B, num_iter, e)

    return v_min


def calculate_homography_math(src_pts, dst_pts):
    A = []
    for i in range(4):
        x, y = src_pts[i]
        x_prime, y_prime = dst_pts[i]

        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

    A = np.array(A)

    AtA = np.dot(A.T, A)

    # Eigenvector of the smallest eigenvalue of A^T A — this is the h that
    # minimizes ||Ah||^2 on the unit sphere, i.e. the flattened homography.
    H_flat = smallest_eigenvector(AtA)

    H = H_flat.reshape(3, 3)

    H = H / H[2, 2]

    return H


def send_speed_to_robot(left_speed, right_speed):
    command = {"L": round(left_speed, 2), "R": round(right_speed, 2)}
    try:
        sock.sendto(json.dumps(command).encode('utf-8'), (ROBOT_IP, UDP_PORT))
    except Exception as e:
        print(f"{e}")


def get_real_coords(u, v):
    if H_matrix is None: return 0.0, 0.0
    pixel_point = np.array([u, v, 1])
    world_point_homo = np.dot(H_matrix, pixel_point)
    r_x = world_point_homo[0] / world_point_homo[2]
    r_y = world_point_homo[1] / world_point_homo[2]
    return r_x, r_y


def autopilot_thread(target_x, target_y, target_angle, angle_error, distance):
    global robot_center, robot_yaw, is_moving

    sec_per_degree = 0.0165
    sec_per_meter = 7.5

    angle_deg = math.degrees(angle_error)
    turn_time = abs(angle_deg) * sec_per_degree
    drive_time = (distance / 1000.0) * sec_per_meter

    if abs(angle_deg) > 5:

        l_speed = 0.25 if angle_error > 0 else -0.25
        r_speed = -0.25 if angle_error > 0 else 0.25

        t_start = time.time()
        while time.time() - t_start < turn_time:
            send_speed_to_robot(l_speed, r_speed)
            time.sleep(0.1)

    if distance > 20:
        t_start = time.time()
        while time.time() - t_start < drive_time:
            send_speed_to_robot(0.3, 0.3)
            time.sleep(0.1)

    send_speed_to_robot(0, 0)

    robot_center = (target_x, target_y)
    robot_yaw = target_angle
    is_moving = False


def mouse_event(event, u, v, flags, param):
    global clicked_points, world_pts, robot_points, H_matrix, calibration_stage
    global current_mouse_pos, real_coords, robot_center, robot_yaw, is_moving

    current_mouse_pos = (u, v)

    if H_matrix is not None:
        real_coords = get_real_coords(u, v)

    if event == cv2.EVENT_LBUTTONDOWN:
        if calibration_stage == 0:
            user_input = simpledialog.askstring("Coordinates Input",
                                                f"Floor: Point {len(clicked_points) + 1}/4\nEnter physical X, Y (mm):")
            if user_input:
                try:
                    parts = user_input.split(',')
                    r_x, r_y = float(parts[0].strip()), float(parts[1].strip())
                    clicked_points.append((u, v))
                    world_pts.append([r_x, r_y])
                    print(f"Floor: point {len(clicked_points)} -> [{r_x}, {r_y}]")

                    if len(clicked_points) == 4:
                        H_matrix = calculate_homography_math(clicked_points, world_pts)
                        calibration_stage = 1
                        print("\n=== FLOOR CALIBRATED ===")
                        print("Click: 1) BACK part, 2) FRONT (nose)!\n")
                except:
                    print("Error occurred while entering coordinates!")

        elif calibration_stage == 1:
            robot_points.append(get_real_coords(u, v))
            if len(robot_points) == 2:
                tail, nose = robot_points[0], robot_points[1]
                robot_center = ((tail[0] + nose[0]) / 2, (tail[1] + nose[1]) / 2)
                robot_yaw = math.atan2(nose[1] - tail[1], nose[0] - tail[0])
                calibration_stage = 2
                print(f"\n=== ROBOT FOUND ===")
                print("Click on the target!\n")

        elif calibration_stage == 2:
            if is_moving:
                print("[WARNING] Robot is still moving! Please wait for it to stop.")
                return

            target_x, target_y = real_coords
            dx = target_x - robot_center[0]
            dy = target_y - robot_center[1]
            distance = math.sqrt(dx ** 2 + dy ** 2)

            target_angle = math.atan2(dy, dx)
            angle_error = target_angle - robot_yaw
            angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

            is_moving = True
            threading.Thread(target=autopilot_thread,
                             args=(target_x, target_y, target_angle, angle_error, distance),
                             daemon=True).start()


def main():
    global clicked_points, world_pts, robot_points, calibration_stage, H_matrix, robot_center, robot_yaw, real_coords, is_moving

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    window_name = "RaspRover Navigation Control"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_event)

    while True:
        ret, frame = cap.read()
        if not ret: break

        if calibration_stage == 0:
            cv2.putText(frame, f"Step 1: Floor ({len(clicked_points)}/4)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)
            for idx, pt in enumerate(clicked_points):
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)

        elif calibration_stage == 1:
            cv2.putText(frame, f"Step 2: TAIL -> NOSE ({len(robot_points)}/2)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 165, 255), 2)
            pts = np.array(clicked_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

        elif calibration_stage == 2:
            status_text = "DRIVING..." if is_moving else "READY. Click target!"
            status_color = (0, 0, 255) if is_moving else (0, 255, 0)
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            pts = np.array(clicked_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

            if robot_center:
                H_inv = np.linalg.inv(H_matrix)
                center_w = np.array([robot_center[0], robot_center[1], 1])
                center_p = np.dot(H_inv, center_w)
                cx, cy = int(center_p[0] / center_p[2]), int(center_p[1] / center_p[2])

                cv2.circle(frame, (cx, cy), 8, (255, 0, 255), -1)

                line_len = 50
                ex = int(cx + line_len * math.cos(robot_yaw))
                ey = int(cy + line_len * math.sin(robot_yaw))
                cv2.line(frame, (cx, cy), (ex, ey), (0, 0, 255), 3)

        cv2.circle(frame, current_mouse_pos, 4, (0, 255, 255), -1)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            send_speed_to_robot(0, 0)
            break
        elif key == ord('s'):
            send_speed_to_robot(0, 0)
            is_moving = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
