FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    autoconf \
    automake \
    libtool \
    pkg-config \
    git \
    libglib2.0-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libplib-dev \
    libopenal-dev \
    libalut-dev \
    libxi-dev \
    libxmu-dev \
    libxrender-dev \
    libxrandr-dev \
    libxxf86vm-dev \
    libpng-dev \
    libvorbis-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone TORCS
RUN git clone https://github.com/fmirus/torcs-1.3.7.git /torcs

WORKDIR /torcs

# Build
RUN CFLAGS="-fPIC" \
    CPPFLAGS="-fPIC" \
    CXXFLAGS="-fPIC" \
    ./configure --prefix=/torcs/BUILD \
    --disable-xrandr \
    --disable-oss \
    && make  \
    && make install \
    && make datainstall

WORKDIR /torcs/BUILD

EXPOSE 3001/udp

# Copying over car configuration files
COPY ./car-config/scr_server.xml /torcs/BUILD/share/games/torcs/drivers/scr_server/scr_server.xml
COPY ./car-config/0 /torcs/BUILD/share/games/torcs/drivers/scr_server/0
COPY ./car-config/1 /torcs/BUILD/share/games/torcs/drivers/scr_server/1

COPY ./practise.xml /torcs/BUILD/share/games/torcs/config/raceman/practice.xml

COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh


# Fix Linux permissions and line endings
RUN chmod +x /usr/local/bin/entrypoint.sh && \
    sed -i 's/\r$//' /usr/local/bin/entrypoint.sh

WORKDIR /torcs/BUILD
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]