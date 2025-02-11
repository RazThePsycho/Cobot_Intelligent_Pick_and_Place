ARG BASE_IMG
# docker build -t soution-img . --build-arg  BASE_IMG=osrf/ros:noetic-desktop-full

FROM ${BASE_IMG}

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND noninteractive

RUN cd /ros_ws && catkin build


ENTRYPOINT [ "/bin/bash", "-ci", " cd /ros_ws && catkin build && source devel/setup.bash && roslaunch solution_master start.launch" ]