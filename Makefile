CUDA_VER=12.1
# ifeq ($(CUDA_VER),)
#   $(error "CUDA_VER is not set")
# endif

DS_VER=6.3
# ifeq ($(DS_VER),)
#   $(error "DS_VER is not set")
# endif

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)
CXX:= g++

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0

SRCS:= gstdscropper.cpp

INCS:= $(wildcard *.h)
LIB:=libnvdsgst_dscropper.so



CFLAGS+= -fPIC -DDS_VERSION=\"$(DS_VER)\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../includes


GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(DS_VER)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(DS_VER)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -ldl \
	-lnppc -lnppig -lnpps -lnppicc -lnppidei \
	-L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform\
	-Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.cpp=.o)

CFLAGS+= -DWITH_OPENCV
PKGS+= opencv4

CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CFLAGS)
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)


install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
