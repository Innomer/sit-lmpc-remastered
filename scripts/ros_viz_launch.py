"""Foxglove visualization for an already running gym bridge (map, car model, websocket relay).

ros2 launch scripts/ros_viz_launch.py map:=Spielberg open_foxglove:=true
Mirrors f1tenth_gym_ros's gym_bridge_launch.py helpers without starting a second bridge.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def setup(context):
    share = get_package_share_directory("f1tenth_gym_ros")
    map_name = LaunchConfiguration("map").perform(context)
    port = LaunchConfiguration("foxglove_port").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() == "true"
    map_yaml = os.path.join(share, "maps", f"{map_name}.yaml")
    description = ParameterValue(Command([
        "xacro ", os.path.join(share, "urdf", "racecar_mesh.xacro"),
        " car_name:=ego_racecar accent_color:=\"0.3 0.57 1.0 1.0\" vehicle:=f1tenth",
    ]), value_type=str)
    return [
        Node(package="foxglove_bridge", executable="foxglove_bridge", name="foxglove_bridge",
             parameters=[{"port": int(port)}], output="screen"),
        Node(package="nav2_map_server", executable="map_server",
             parameters=[{"yaml_filename": map_yaml}, {"topic": "map"}, {"frame_id": "map"},
                         {"use_sim_time": use_sim_time}]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_localization",
             parameters=[{"use_sim_time": use_sim_time}, {"autostart": True},
                         {"node_names": ["map_server"]}]),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             name="ego_robot_state_publisher",
             parameters=[{"robot_description": description}, {"use_sim_time": use_sim_time}],
             remappings=[("/robot_description", "ego_robot_description")]),
        ExecuteProcess(
            cmd=["xdg-open", f"https://app.foxglove.dev/?ds=foxglove-websocket&ds.url=ws://localhost:{port}"],
            condition=IfCondition(LaunchConfiguration("open_foxglove")),
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("map", default_value="Spielberg"),
        DeclareLaunchArgument("foxglove_port", default_value="8765"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("open_foxglove", default_value="true"),
        OpaqueFunction(function=setup),
    ])
