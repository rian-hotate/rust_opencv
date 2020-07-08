FROM rust:latest

RUN apt-get update && apt-get install -y vim build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

RUN mkdir /work

WORKDIR work

RUN git clone https://github.com/opencv/opencv.git

RUN git clone https://github.com/opencv/opencv_contrib.git

RUN mkdir /work/opencv/build

WORKDIR /work/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..o

RUN make -j3

RUN make install

RUN echo "alias ll='ls -la --color'">>~/.bashrc

RUN rustup nightly

RUN rustup default nightly

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib
