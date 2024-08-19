#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf.transformations as transformations
import tf
from geometry_msgs.msg import TransformStamped
import time
import tf2_ros
import math

frame = None
d1 = []
d2 = []
prev_gray = np.array([])
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
prev_angle = 0
image_size=(1500, 1500)
scale=1000
line_thickness=3
quat = [0,0,0,1]
loc = [0,0,0] # x, y, theta
# Başlangıç koordinatları
x, y = image_size[1] // 2, image_size[0] // 2
# Boş bir görüntü oluştur
image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

last_call = None

def create_transformation_matrix(translation, rotation):
    """
    translation: [x, y, z]
    rotation: [roll, pitch, yaw] in radians
    """
    tx, ty, tz = translation
    roll, pitch, yaw = rotation

    # Dönüşüm matrislerini oluştur
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    # Toplam dönüşüm matrisi
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [tx, ty, tz]
    
    return T

T_base_camera = create_transformation_matrix([0.0, -0.01, 0.15], [0, 1.57, 0])

def image_callback(msg):
    global d1, d2, frame, prev_gray, feature_params
    global prev_angle, scale, line_thickness, image, x, y
    global last_call
    global T_base_camera
    global quat
    global loc

    timer = time.time()
    if last_call is not None:
        interval = timer - last_call
        fps = 1 / interval
        last_call = timer
        #print("FPS", fps)

    if last_call is None:
        last_call = timer    

    if frame is None:
        frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #prev_gray = prev_gray[:, 280:1000]
        prev_gray = prev_gray[60:660, 397:997]
        return

    lk_params = dict(winSize=(120, 120), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    frame = CvBridge().imgmsg_to_cv2(msg, "bgr8")
    #frame = frame[:, 280:1000]
    frame = frame[60:660, 397:997]
    #print(frame.shape)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    # Optik akışı hesapla
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Optik akış vektörlerini çiz
    temp_d1s = []
    temp_d2s = []
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        temp_d1s.append(a - c)
        temp_d2s.append(b - d)
        # Hareket vektörlerini çiz
        frame = cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
        # Özellik noktalarını çiz
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
    #temp_d1 = np.median(np.array(temp_d1)) * 0.0001431767459402224
    #temp_d2 = np.median(np.array(temp_d2)) * 0.0001508210276773436
    temp_d1 = np.median(np.array(temp_d1s)) * 0.000096954
    temp_d2 = np.median(np.array(temp_d2s)) * 0.000096954
    #print(f"d1: %.7f, d2: %.7f" %(temp_d1,temp_d2))
    #temp_d1 = 0
    d1.append(temp_d1)
    d2.append(temp_d2)
    loc[0] = loc[0] + temp_d1
    loc[1] = loc[1] + temp_d2

    vel = np.sqrt(temp_d1**2 + temp_d2**2) * fps
    #print(f"vel {vel:.2f} m/s")

    if np.isnan(temp_d1) or np.isnan(temp_d2) or np.isnan(x + temp_d1 * scale) or np.isnan(y + temp_d2 * scale):
        return
    
    # Açı hesabı
    
    try:
        H, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
        theta = np.arctan2(H[1, 0], H[0, 0])
        #print(H[1, 0], H[0, 0])
        loc[2] = loc[2] + theta
        print(f"Açısal değişim: {np.degrees(loc[2])} derece")
    except:
        print("Not enough points to calculate angle")


    # Hareket vektörünün büyüklüğünü ve açısını hesapla
    magnitude = np.sqrt(temp_d1**2 + temp_d2**2)
    angle = np.arctan2(temp_d2, temp_d1) - np.pi/2
    #angle = theta

    if magnitude > 0:
        #print(np.degrees(angle),magnitude)
        T_camera_new_old = create_transformation_matrix([temp_d1, temp_d2, 0], [0, 0, loc[2]])

        angle += prev_angle
        # Yeni noktanın konumunu hesapla
        #print("tot. angle", np.degrees(angle))
        new_x = x + magnitude * np.cos(angle) * scale
        new_y = y + magnitude * np.sin(angle) * scale

        T_base_camera_new = np.dot(T_base_camera, T_camera_new_old)
        T_base_new = np.dot(np.linalg.inv(T_base_camera), T_base_camera_new)
        new_position = T_base_new[0:3, 3]
        new_orientation = [np.arctan2(T_base_new[2, 1], T_base_new[2, 2]),
                        np.arctan2(-T_base_new[2, 0], np.sqrt(T_base_new[2, 1]**2 + T_base_new[2, 2]**2)),
                        np.arctan2(T_base_new[1, 0], T_base_new[0, 0])]

        #print("Base_link'in yeni pozisyonu:", new_position)
        #print("Base_link'in yeni oryantasyonu (roll, pitch, yaw):", new_orientation)   

        if abs(temp_d1) >= 0.001 or abs(temp_d2) >= 0.001:
            quat = tf.transformations.quaternion_from_matrix(T_base_new)

        tf_pub = tf.TransformBroadcaster()

        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = "map"
        tf_msg.child_frame_id = "base_estimation"
        """tf_msg.transform.translation.x = new_x/1000 - 0.78 + 0.95
        tf_msg.transform.translation.y = new_y/1000 + 1.22 - 0.01
        tf_msg.transform.translation.z = 0 + 0.1"""
        tf_msg.transform.translation.x = loc[1] - 0.78 + 0.95
        tf_msg.transform.translation.y = loc[0] + 1.22 - 0.01
        tf_msg.transform.translation.z = 0 + 0.1
        """tf_msg.transform.translation.x = new_x/1000
        tf_msg.transform.translation.y = new_y/1000
        tf_msg.transform.translation.z = 0"""
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]

        tf_pub.sendTransformMessage(tf_msg)

        # Hareket yönünü gösteren ok çiz
        #cv2.arrowedLine(image, (int(x), int(y)), (int(new_x), int(new_y)), (0, 0, 255), line_thickness)
        # cv2.circle(image, (int(1000*new_position[0])+640, int(1000*new_position[1])+360), 3, (255, 0, 0), -1)
        # Yeni noktayı güncelle
        x, y = new_x, new_y
        # Önceki hareket açısını güncelle
        prev_angle = angle
        

    cv2.imshow('Optical flow', frame)
    #cv2.imshow('Directions', image)
    cv2.waitKey(1)
    prev_gray = gray
    

def image_subscriber():
    rospy.init_node('vslam_optical_flow_node', anonymous=True)

    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        image_subscriber()
    except rospy.ROSInterruptException:
        pass
