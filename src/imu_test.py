#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped
import message_filters

def imu_callback(accel_data, gyro_data):
    imu_msg = Imu()
    imu_msg.header.stamp = rospy.Time.now()
    imu_msg.header.frame_id = "imu_link"

    # Accelerometer data
    imu_msg.linear_acceleration.x = accel_data.vector.x
    imu_msg.linear_acceleration.y = accel_data.vector.y
    imu_msg.linear_acceleration.z = accel_data.vector.z

    # Gyroscope data
    imu_msg.angular_velocity.x = gyro_data.vector.x
    imu_msg.angular_velocity.y = gyro_data.vector.y
    imu_msg.angular_velocity.z = gyro_data.vector.z

    imu_publisher.publish(imu_msg)

if __name__ == '__main__':
    rospy.init_node('imu_data_combiner')

    imu_publisher = rospy.Publisher('/imu/data_raw', Imu, queue_size=10)

    accel_sub = message_filters.Subscriber('/imu/acceleration', Vector3Stamped)
    gyro_sub = message_filters.Subscriber('/imu/angular_velocity', Vector3Stamped)

    ts = message_filters.TimeSynchronizer([accel_sub, gyro_sub], 10)
    ts.registerCallback(imu_callback)

    rospy.spin()
