FROM pytorch/pytorch:latest
ENV DEBIAN_FRONTEND noninteractive

# install dependencies via apt
ENV DEBCONF_NOWARNINGS yes
RUN set -x && \
  apt-get update -y -qq && \
  apt-get upgrade -y -qq --no-install-recommends && \
  : "basic dependencies" && \
  apt-get install -y -qq \
    build-essential \
    pkg-config \
    cmake \
    git \
    vim \
    wget \
    curl \
    tar \
    ca-certificates \
    unzip && \
  # : "OpenCV dependencies" && \
  # apt-get install -y -qq \
  #   libgtk-3-dev \
  #   libjpeg-dev \
  #   libpng++-dev \
  #   libtiff-dev \
  #   libopenexr-dev \
  #   libwebp-dev \
  #   ffmpeg \
  #   libavcodec-dev \
  #   libavformat-dev \
  #   libavutil-dev \
  #   libswscale-dev \
  #   libavresample-dev && \
  : "other dependencies" && \
  apt-get install -y -qq \
    libyaml-cpp-dev && \
  : "remove cache" && \
  apt-get autoremove -y -qq && \
  rm -rf /var/lib/apt/lists/*

ARG CMAKE_INSTALL_PREFIX=/usr/local
ARG NUM_THREADS=2

ENV CPATH=${CMAKE_INSTALL_PREFIX}/include:${CPATH}
ENV C_INCLUDE_PATH=${CMAKE_INSTALL_PREFIX}/include:${C_INCLUDE_PATH}
ENV CPLUS_INCLUDE_PATH=${CMAKE_INSTALL_PREFIX}/include:${CPLUS_INCLUDE_PATH}
ENV LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH}

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# OpenCV
# ARG OPENCV_VERSION=4.1.0
# WORKDIR /tmp
# RUN set -x && \
#   wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
#   unzip -q ${OPENCV_VERSION}.zip && \
#   rm -rf ${OPENCV_VERSION}.zip && \
#   cd opencv-${OPENCV_VERSION} && \
#   mkdir -p build && \
#   cd build && \
#   cmake \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
#     -DBUILD_DOCS=OFF \
#     -DBUILD_EXAMPLES=OFF \
#     -DBUILD_JASPER=OFF \
#     -DBUILD_OPENEXR=OFF \
#     -DBUILD_PERF_TESTS=OFF \
#     -DBUILD_TESTS=OFF \
#     -DBUILD_opencv_apps=OFF \
#     -DBUILD_opencv_dnn=OFF \
#     -DBUILD_opencv_ml=OFF \
#     -DBUILD_opencv_python_bindings_generator=OFF \
#     -DENABLE_CXX11=ON \
#     -DENABLE_FAST_MATH=ON \
#     -DWITH_EIGEN=ON \
#     -DWITH_FFMPEG=ON \
#     -DWITH_OPENMP=ON \
#     .. && \
#   make -j${NUM_THREADS} && \
#   make install && \
#   cd /tmp && \
#   rm -rf *
# ENV OpenCV_DIR=${CMAKE_INSTALL_PREFIX}/lib/cmake/opencv4

COPY ./requirements.txt /requirements.txt
COPY ./.bash_prompt /root/.bash_prompt
COPY ./.bashrc /root/.bashrc

RUN pip3 install --no-cache-dir -r /requirements.txt

WORKDIR /home/saqibcephfspvc
ENTRYPOINT ["/bin/bash"]