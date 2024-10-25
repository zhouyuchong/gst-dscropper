/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


#include <string.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>

#include "gstdsclipper.h"

#include <sys/time.h>
#include <condition_variable>
#include <mutex>
#include <thread>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

GST_DEBUG_CATEGORY_STATIC (gst_dsclipper_debug);
#define GST_CAT_DEFAULT gst_dsclipper_debug
#define USE_EGLIMAGE 1


static GQuark _dsmeta_quark = 0;

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_PROCESSING_WIDTH,
  PROP_PROCESSING_HEIGHT,
  PROP_PROCESS_FULL_FRAME,
  PROP_BATCH_SIZE,
  PROP_GPU_DEVICE_ID
};

#define CHECK_NVDS_MEMORY_AND_GPUID(object, surface)  \
  ({ int _errtype=0;\
   do {  \
    if ((surface->memType == NVBUF_MEM_DEFAULT || surface->memType == NVBUF_MEM_CUDA_DEVICE) && \
        (surface->gpuId != object->gpu_id))  { \
    GST_ELEMENT_ERROR (object, RESOURCE, FAILED, \
        ("Input surface gpu-id doesnt match with configured gpu-id for element," \
         " please allocate input using unified memory, or use same gpu-ids"),\
        ("surface-gpu-id=%d,%s-gpu-id=%d",surface->gpuId,GST_ELEMENT_NAME(object),\
         object->gpu_id)); \
    _errtype = 1;\
    } \
    } while(0); \
    _errtype; \
  })


/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_PROCESSING_WIDTH 640
#define DEFAULT_PROCESSING_HEIGHT 480
#define DEFAULT_PROCESS_FULL_FRAME TRUE
#define DEFAULT_GPU_ID 0
#define DEFAULT_BATCH_SIZE 1

#define RGB_BYTES_PER_PIXEL 3
#define RGBA_BYTES_PER_PIXEL 4
#define Y_BYTES_PER_PIXEL 1
#define UV_BYTES_PER_PIXEL 2

#define MIN_INPUT_OBJECT_WIDTH 16
#define MIN_INPUT_OBJECT_HEIGHT 16

#define MAX_QUEUE_SIZE 20

#define CHECK_NPP_STATUS(npp_status,error_str) do { \
  if ((npp_status) != NPP_SUCCESS) { \
    g_print ("Error: %s in %s at line %d: NPP Error %d\n", \
        error_str, __FILE__, __LINE__, npp_status); \
    goto error; \
  } \
} while (0)

#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
    goto error; \
  } \
} while (0)

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_dsclipper_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_dsclipper_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_dsclipper_parent_class parent_class
G_DEFINE_TYPE (GstDsClipper, gst_dsclipper, GST_TYPE_BASE_TRANSFORM);

static void gst_dsclipper_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_dsclipper_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_dsclipper_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_dsclipper_start (GstBaseTransform * btrans);
static gboolean gst_dsclipper_stop (GstBaseTransform * btrans);

static GstFlowReturn
gst_dsclipper_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf);
static GstFlowReturn
gst_dsclipper_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf);

static void
attach_metadata_full_frame (GstDsClipper * dsclipper,
    NvDsFrameMeta * frame_meta, gdouble scale_ratio, DsClipperOutput * output,
    guint batch_id);
static void attach_metadata_object (GstDsClipper * dsclipper,
    NvDsObjectMeta * obj_meta, DsClipperOutput * output);

static gpointer gst_dsclipper_output_loop (gpointer data);

static gpointer gst_dsclipper_data_loop (gpointer data);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_dsclipper_class_init (GstDsClipperClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  // Indicates we want to use DS buf api
  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_dsclipper_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_dsclipper_get_property);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_dsclipper_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_dsclipper_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_dsclipper_stop);

  gstbasetransform_class->submit_input_buffer =
      GST_DEBUG_FUNCPTR (gst_dsclipper_submit_input_buffer);
  gstbasetransform_class->generate_output =
      GST_DEBUG_FUNCPTR (gst_dsclipper_generate_output);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id",
          "Unique ID",
          "Unique ID for the element. Can be used to identify output of the"
          " element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_WIDTH,
      g_param_spec_int ("processing-width",
          "Processing Width",
          "Width of the input buffer to algorithm",
          1, G_MAXINT, DEFAULT_PROCESSING_WIDTH, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_HEIGHT,
      g_param_spec_int ("processing-height",
          "Processing Height",
          "Height of the input buffer to algorithm",
          1, G_MAXINT, DEFAULT_PROCESSING_HEIGHT, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESS_FULL_FRAME,
      g_param_spec_boolean ("full-frame",
          "Full frame",
          "Enable to process full frame or disable to process objects detected"
          "by primary detector", DEFAULT_PROCESS_FULL_FRAME, (GParamFlags)
          (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
      g_param_spec_uint ("batch-size", "Batch Size",
          "Maximum batch size for processing",
          1, NVDSCLIPPER_MAX_BATCH_SIZE, DEFAULT_BATCH_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id",
          "Set GPU Device ID",
          "Set GPU Device ID", 0,
          G_MAXUINT, 0,
          GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));
  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsclipper_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsclipper_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "DsClipper plugin",
      "DsClipper Plugin",
      "Process a 3rdparty example algorithm on objects / full frame",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");
}

static void
gst_dsclipper_init (GstDsClipper * dsclipper)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (dsclipper);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  /* Initialize all property variables to default values */
  dsclipper->unique_id = DEFAULT_UNIQUE_ID;
  dsclipper->processing_width = DEFAULT_PROCESSING_WIDTH;
  dsclipper->processing_height = DEFAULT_PROCESSING_HEIGHT;
  dsclipper->process_full_frame = DEFAULT_PROCESS_FULL_FRAME;
  dsclipper->gpu_id = DEFAULT_GPU_ID;
  dsclipper->max_batch_size = DEFAULT_BATCH_SIZE;
  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_dsclipper_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsClipper *dsclipper = GST_DSCLIPPER (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      dsclipper->unique_id = g_value_get_uint (value);
      break;
    case PROP_PROCESSING_WIDTH:
      dsclipper->processing_width = g_value_get_int (value);
      break;
    case PROP_PROCESSING_HEIGHT:
      dsclipper->processing_height = g_value_get_int (value);
      break;
    case PROP_PROCESS_FULL_FRAME:
      dsclipper->process_full_frame = g_value_get_boolean (value);
      break;
    case PROP_GPU_DEVICE_ID:
      dsclipper->gpu_id = g_value_get_uint (value);
      break;
    case PROP_BATCH_SIZE:
      dsclipper->max_batch_size = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_dsclipper_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsClipper *dsclipper = GST_DSCLIPPER (object);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, dsclipper->unique_id);
      break;
    case PROP_PROCESSING_WIDTH:
      g_value_set_int (value, dsclipper->processing_width);
      break;
    case PROP_PROCESSING_HEIGHT:
      g_value_set_int (value, dsclipper->processing_height);
      break;
    case PROP_PROCESS_FULL_FRAME:
      g_value_set_boolean (value, dsclipper->process_full_frame);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, dsclipper->gpu_id);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, dsclipper->max_batch_size);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Initialize all resources and start the process thread
 */
static gboolean
gst_dsclipper_start (GstBaseTransform * btrans)
{
  GstDsClipper *dsclipper = GST_DSCLIPPER (btrans);
  std::string nvtx_str;
#ifdef WITH_OPENCV
  // OpenCV mat containing RGB data
  cv::Mat * cvmat;
#else
  NvBufSurface * inter_buf;
#endif
  NvBufSurfaceCreateParams create_params;
  DsClipperInitParams init_params =
      { dsclipper->processing_width, dsclipper->processing_height,
    dsclipper->process_full_frame };

  /* Algorithm specific initializations and resource allocation. */
  dsclipper->dsclipperlib_ctx = DsClipperCtxInit (&init_params);

  GST_DEBUG_OBJECT (dsclipper, "ctx lib %p \n", dsclipper->dsclipperlib_ctx);

  nvtx_str = "GstNvDsClipper: UID=" + std::to_string(dsclipper->unique_id);
  auto nvtx_deleter = [](nvtxDomainHandle_t d) { nvtxDomainDestroy (d); };
  std::unique_ptr<nvtxDomainRegistration, decltype(nvtx_deleter)> nvtx_domain_ptr (
      nvtxDomainCreate(nvtx_str.c_str()), nvtx_deleter);

  CHECK_CUDA_STATUS (cudaSetDevice (dsclipper->gpu_id),
      "Unable to set cuda device");

  CHECK_CUDA_STATUS (cudaStreamCreate (&dsclipper->cuda_stream),
      "Could not create cuda stream");

#ifdef WITH_OPENCV
  if (dsclipper->inter_buf)
    NvBufSurfaceDestroy (dsclipper->inter_buf);
  dsclipper->inter_buf = NULL;
#endif

  /* An intermediate buffer for NV12/RGBA to BGR conversion  will be
   * required. Can be skipped if custom algorithm can work directly on NV12/RGBA. */
  create_params.gpuId = dsclipper->gpu_id;
  create_params.width = dsclipper->processing_width;
  create_params.height = dsclipper->processing_height;
  create_params.size = 0;
  create_params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
  create_params.layout = NVBUF_LAYOUT_PITCH;
#ifdef __aarch64__
  create_params.memType = NVBUF_MEM_DEFAULT;
#else
  create_params.memType = NVBUF_MEM_CUDA_UNIFIED;
#endif

#ifdef WITH_OPENCV
  if (NvBufSurfaceCreate (&dsclipper->inter_buf, dsclipper->max_batch_size,
          &create_params) != 0) {
    GST_ERROR ("Error: Could not allocate internal buffer for dsclipper");
    goto error;
  }
#endif

  /* Create process queue and cvmat queue to transfer data between threads.
   * We will be using this queue to maintain the list of frames/objects
   * currently given to the algorithm for processing. */
  dsclipper->process_queue = g_queue_new ();
  dsclipper->buf_queue = g_queue_new ();
  dsclipper->data_queue = g_queue_new ();

#ifdef WITH_OPENCV
  /* Push cvmat buffer twice on the buf_queue which will handle the
   * different processing speed between input thread and process thread
   * cvmat queue is used for getting processed data from the process thread*/
  for (int i = 0; i < 2; i++) {
    // CV Mat containing interleaved RGB data.
    cvmat = new cv::Mat[dsclipper->max_batch_size];

    for (guint j = 0; j < dsclipper->max_batch_size; j++) {
      cvmat[j] =
          cv::Mat (dsclipper->processing_height, dsclipper->processing_width,
          CV_8UC3);
    }

    if (!cvmat)
      goto error;

    g_queue_push_tail (dsclipper->buf_queue, cvmat);
  }

  GST_DEBUG_OBJECT (dsclipper, "created CV Mat\n");
#else
  for (int i = 0; i < 2; i++) {
    if (NvBufSurfaceCreate (&inter_buf, dsclipper->max_batch_size,
          &create_params) != 0) {
      GST_ERROR ("Error: Could not allocate internal buffer for dsclipper");
      goto error;
    }

    g_queue_push_tail (dsclipper->buf_queue, inter_buf);
  }
#endif

  /* Set the NvBufSurfTransform config parameters. */
  dsclipper->transform_config_params.compute_mode =
      NvBufSurfTransformCompute_Default;
  dsclipper->transform_config_params.gpu_id = dsclipper->gpu_id;

  /* Create the intermediate NvBufSurface structure for holding an array of input
   * NvBufSurfaceParams for batched transforms. */
  dsclipper->batch_insurf.surfaceList =
      new NvBufSurfaceParams[dsclipper->max_batch_size];
  dsclipper->batch_insurf.batchSize = dsclipper->max_batch_size;
  dsclipper->batch_insurf.gpuId = dsclipper->gpu_id;

  /* Set up the NvBufSurfTransformParams structure for batched transforms. */
  dsclipper->transform_params.src_rect =
      new NvBufSurfTransformRect[dsclipper->max_batch_size];
  dsclipper->transform_params.dst_rect =
      new NvBufSurfTransformRect[dsclipper->max_batch_size];
  dsclipper->transform_params.transform_flag =
      NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC |
      NVBUFSURF_TRANSFORM_CROP_DST;
  dsclipper->transform_params.transform_flip = NvBufSurfTransform_None;
  dsclipper->transform_params.transform_filter =
      NvBufSurfTransformInter_Default;

  /* Start a thread which will pop output from the algorithm, form NvDsMeta and
   * push buffers to the next element. */
  dsclipper->process_thread =
      g_thread_new ("dsclipper-process-thread", gst_dsclipper_output_loop,
      dsclipper);
  dsclipper->data_thread =
      g_thread_new ("dsclipper-data-thread", gst_dsclipper_data_loop,
      dsclipper);
  dsclipper->nvtx_domain = nvtx_domain_ptr.release ();

  return TRUE;
error:

  delete[]dsclipper->transform_params.src_rect;
  delete[]dsclipper->transform_params.dst_rect;
  delete[]dsclipper->batch_insurf.surfaceList;

  if (dsclipper->cuda_stream) {
    cudaStreamDestroy (dsclipper->cuda_stream);
    dsclipper->cuda_stream = NULL;
  }
  if (dsclipper->dsclipperlib_ctx)
    DsClipperCtxDeinit (dsclipper->dsclipperlib_ctx);
  return FALSE;
}

/**
 * Stop the process thread and free up all the resources
 */
static gboolean
gst_dsclipper_stop (GstBaseTransform * btrans)
{
  GstDsClipper *dsclipper = GST_DSCLIPPER (btrans);

#ifdef WITH_OPENCV
  cv::Mat * cvmat;
#else
  NvBufSurface * inter_buf;
#endif

  g_mutex_lock (&dsclipper->process_lock);

  /* Wait till all the items in the queue are handled. */
  while (!g_queue_is_empty (dsclipper->process_queue)) {
    g_cond_wait (&dsclipper->process_cond, &dsclipper->process_lock);
  }

  g_mutex_lock (&dsclipper->data_lock);
  while (!g_queue_is_empty (dsclipper->data_queue)) {
    g_cond_wait (&dsclipper->data_cond, &dsclipper->data_lock);
  }

#ifdef WITH_OPENCV
  while (!g_queue_is_empty (dsclipper->buf_queue)) {
    cvmat = (cv::Mat *) g_queue_pop_head (dsclipper->buf_queue);
    delete[]cvmat;
    cvmat = NULL;
  }
#else
  while (!g_queue_is_empty (dsclipper->buf_queue)) {
    inter_buf = (NvBufSurface *) g_queue_pop_head (dsclipper->buf_queue);
    if (inter_buf)
      NvBufSurfaceDestroy (inter_buf);
    inter_buf = NULL;
  }
#endif
  dsclipper->stop = TRUE;

  g_cond_broadcast (&dsclipper->process_cond);
  g_mutex_unlock (&dsclipper->process_lock);

  g_cond_broadcast (&dsclipper->data_cond);
  g_mutex_unlock (&dsclipper->data_lock);

  g_thread_join (dsclipper->process_thread);
  g_thread_join (dsclipper->data_thread);

#ifdef WITH_OPENCV
  if (dsclipper->inter_buf)
    NvBufSurfaceDestroy (dsclipper->inter_buf);
  dsclipper->inter_buf = NULL;
#endif

  if (dsclipper->cuda_stream)
    cudaStreamDestroy (dsclipper->cuda_stream);
  dsclipper->cuda_stream = NULL;

  delete[]dsclipper->transform_params.src_rect;
  delete[]dsclipper->transform_params.dst_rect;
  delete[]dsclipper->batch_insurf.surfaceList;

#ifdef WITH_OPENCV
  GST_DEBUG_OBJECT (dsclipper, "deleted CV Mat \n");
#endif

  // Deinit the algorithm library
  DsClipperCtxDeinit (dsclipper->dsclipperlib_ctx);
  dsclipper->dsclipperlib_ctx = NULL;

  GST_DEBUG_OBJECT (dsclipper, "ctx lib released \n");

  g_queue_free (dsclipper->process_queue);
  g_queue_free (dsclipper->data_queue);

  g_queue_free (dsclipper->buf_queue);
  
  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_dsclipper_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstDsClipper *dsclipper = GST_DSCLIPPER (btrans);
  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps (&dsclipper->video_info, incaps);

  CHECK_CUDA_STATUS (cudaSetDevice (dsclipper->gpu_id),
      "Unable to set cuda device");

  return TRUE;

error:
  return FALSE;
}

/**
 * Scale the entire frame to the processing resolution maintaining aspect ratio.
 * Or crop and scale objects to the processing resolution maintaining the aspect
 * ratio and fills data for batched conversation */
static GstFlowReturn
scale_and_fill_data(GstDsClipper * dsclipper,
    NvBufSurfaceParams * src_frame, NvOSD_RectParams * crop_rect_params,
    gdouble & ratio, gint input_width, gint input_height)
{

  gint src_left = GST_ROUND_UP_2((unsigned int)crop_rect_params->left);
  gint src_top = GST_ROUND_UP_2((unsigned int)crop_rect_params->top);
  gint src_width = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->width);
  gint src_height = GST_ROUND_DOWN_2((unsigned int)crop_rect_params->height);

  // Maintain aspect ratio
  double hdest = dsclipper->processing_width * src_height / (double) src_width;
  double wdest = dsclipper->processing_height * src_width / (double) src_height;
  guint dest_width, dest_height;

  if (hdest <= dsclipper->processing_height) {
    dest_width = dsclipper->processing_width;
    dest_height = hdest;
  } else {
    dest_width = wdest;
    dest_height = dsclipper->processing_height;
  }

  // Calculate scaling ratio while maintaining aspect ratio
  ratio = MIN (1.0 * dest_width / src_width, 1.0 * dest_height / src_height);

  if ((crop_rect_params->width == 0) || (crop_rect_params->height == 0)) {
    GST_ELEMENT_ERROR (dsclipper, STREAM, FAILED,
        ("%s:crop_rect_params dimensions are zero", __func__), (NULL));
    return GST_FLOW_ERROR;
  }
#ifdef __aarch64__
  if (ratio <= 1.0 / 16 || ratio >= 16.0) {
    // Currently cannot scale by ratio > 16 or < 1/16 for Jetson
    return GST_FLOW_ERROR;
  }
#endif

  /* We will first convert only the Region of Interest (the entire frame or the
   * object bounding box) to RGB and then scale the converted RGB frame to
   * processing resolution. */
  GST_DEBUG_OBJECT (dsclipper, "Scaling and converting input buffer\n");

  /* Create temporary src and dest surfaces for NvBufSurfTransform API. */
  dsclipper->batch_insurf.surfaceList[dsclipper->batch_insurf.numFilled] = *src_frame;

  /* Set the source ROI. Could be entire frame or an object. */
  dsclipper->transform_params.src_rect[dsclipper->batch_insurf.numFilled] = {
  (guint) src_top, (guint) src_left, (guint) src_width, (guint) src_height};
  /* Set the dest ROI. Could be the entire destination frame or part of it to
   * maintain aspect ratio. */
  dsclipper->transform_params.dst_rect[dsclipper->batch_insurf.numFilled] = {
  0, 0, dest_width, dest_height};

  dsclipper->batch_insurf.numFilled++;

  return GST_FLOW_OK;
}

static gboolean
convert_batch_and_push_to_process_thread (GstDsClipper * dsclipper,
    GstDsClipperBatch * batch)
{

  NvBufSurfTransform_Error err;
  NvBufSurfTransformConfigParams transform_config_params;
  std::string nvtx_str;
#ifdef WITH_OPENCV
  cv::Mat in_mat;
#endif

  // Configure transform session parameters for the transformation
  transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
  transform_config_params.gpu_id = dsclipper->gpu_id;
  transform_config_params.cuda_stream = dsclipper->cuda_stream;

  err = NvBufSurfTransformSetSessionParams (&transform_config_params);
  if (err != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (dsclipper, STREAM, FAILED,
        ("NvBufSurfTransformSetSessionParams failed with error %d", err),
        (NULL));
    return FALSE;
  }

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "convert_buf batch_num=" + std::to_string(dsclipper->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();

  nvtxDomainRangePushEx(dsclipper->nvtx_domain, &eventAttrib);

  g_mutex_lock (&dsclipper->process_lock);

  /* Wait if buf queue is empty. */
  while (g_queue_is_empty (dsclipper->buf_queue)) {
    g_cond_wait (&dsclipper->buf_cond, &dsclipper->process_lock);
  }

#ifdef WITH_OPENCV
  /* Pop a buffer from the element's buf queue. */
  batch->cvmat = (cv::Mat *) g_queue_pop_head (dsclipper->buf_queue);
#else
  /* Pop a buffer from the element's buf queue. */
  batch->inter_buf = (NvBufSurface *) g_queue_pop_head (dsclipper->buf_queue);
  dsclipper->inter_buf = batch->inter_buf;
#endif

  g_mutex_unlock (&dsclipper->process_lock);

  //Memset the memory
  for (uint i = 0; i < dsclipper->batch_insurf.numFilled; i++)
    NvBufSurfaceMemSet (dsclipper->inter_buf, i, 0, 0);

  printf("convert func, frame size: %d\n", dsclipper->batch_insurf.numFilled);

  /* Batched tranformation. */
  err = NvBufSurfTransform (&dsclipper->batch_insurf, dsclipper->inter_buf,
      &dsclipper->transform_params);

  nvtxDomainRangePop (dsclipper->nvtx_domain);

  if (err != NvBufSurfTransformError_Success) {
    GST_ELEMENT_ERROR (dsclipper, STREAM, FAILED,
        ("NvBufSurfTransform failed with error %d while converting buffer",
            err), (NULL));
    return FALSE;
  }

   /* Push the batch info structure in the processing queue and notify the process
   * thread that a new batch has been queued. */
  g_mutex_lock (&dsclipper->process_lock);

  g_queue_push_tail (dsclipper->process_queue, batch);
  g_cond_broadcast (&dsclipper->process_cond);

  g_mutex_unlock (&dsclipper->process_lock);

  g_mutex_lock (&dsclipper->data_lock);

  g_queue_push_tail (dsclipper->data_queue, batch);
  g_cond_broadcast (&dsclipper->data_cond);

  g_mutex_unlock (&dsclipper->data_lock);

  return TRUE;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_dsclipper_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf)
{
  GstDsClipper *dsclipper = GST_DSCLIPPER (btrans);
  GstMapInfo in_map_info;
  NvBufSurface *in_surf;
  GstDsClipperBatch *buf_push_batch;
  GstFlowReturn flow_ret;
  std::string nvtx_str;
  std::unique_ptr < GstDsClipperBatch > batch = nullptr;

  NvDsBatchMeta *batch_meta = NULL;
  guint i = 0;
  gdouble scale_ratio = 1.0;
  guint num_filled = 0;

  dsclipper->current_batch_num++;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "buffer_process batch_num=" + std::to_string(dsclipper->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();
  nvtxRangeId_t buf_process_range = nvtxDomainRangeStartEx(dsclipper->nvtx_domain, &eventAttrib);

  memset (&in_map_info, 0, sizeof (in_map_info));

  /* Map the buffer contents and get the pointer to NvBufSurface. */
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    GST_ELEMENT_ERROR (dsclipper, STREAM, FAILED,
        ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__), (NULL));
    return GST_FLOW_ERROR;
  }
  in_surf = (NvBufSurface *) in_map_info.data;

  nvds_set_input_system_timestamp (inbuf, GST_ELEMENT_NAME (dsclipper));

  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (dsclipper, STREAM, FAILED,
        ("NvDsBatchMeta not found for input buffer."), (NULL));
    return GST_FLOW_ERROR;
  }
  num_filled = batch_meta->num_frames_in_batch;


  NvDsFrameMeta *frame_meta = NULL;
  NvDsMetaList *l_frame = NULL;
  NvDsObjectMeta *obj_meta = NULL;
  NvDsMetaList *l_obj = NULL;
  NppStatus stat;
  

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    frame_meta = (NvDsFrameMeta *) (l_frame->data);
    void *host_ptr = NULL;
    void *rgb_frame_ptr = NULL;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);

      /* Should not process on objects smaller than MIN_INPUT_OBJECT_WIDTH x MIN_INPUT_OBJECT_HEIGHT
        * since it will cause hardware scaling issues. */
      if (obj_meta->rect_params.width < MIN_INPUT_OBJECT_WIDTH ||
          obj_meta->rect_params.height < MIN_INPUT_OBJECT_HEIGHT)
        continue;

      if (obj_meta->class_id == 0 && g_queue_get_length(dsclipper->data_queue) < MAX_QUEUE_SIZE) {
        NvBufSurfaceParams * currentFrameParams = in_surf->surfaceList + frame_meta->batch_id;
        NvBufSurfaceColorFormat colorFormat = currentFrameParams->colorFormat;
        size_t framePitch = currentFrameParams->pitch;
        size_t frameSize = currentFrameParams->dataSize;
        uint frameWidth = currentFrameParams->width;
        uint frameHeight = currentFrameParams->height;

        HostFrameInfo* info = g_new(HostFrameInfo, 1);


        // we should transform frame to RGB here while it is on device
        
        // no need since it is already in RGB format
        if (colorFormat < 29 && colorFormat > 19) {
          info->dataSize = currentFrameParams->dataSize;
          info->width = frameWidth;
          info->height = frameHeight;
          info->colorFormat = colorFormat;
          info->pitch = framePitch;
          cudaMallocHost(&host_ptr, frameSize);
          cudaMemcpy((void *)host_ptr,
                  (void *)currentFrameParams->dataPtr,
                  frameSize,
                  cudaMemcpyDeviceToHost); 
        } else {
          const Npp8u* y_plane;                // Pointer to the Y plane
          const Npp8u* uv_plane;               // Pointer to the UV plane
          y_plane = static_cast<const Npp8u*>(currentFrameParams->dataPtr);
          uv_plane = y_plane + (framePitch * frameHeight);
          const Npp8u* pSrc[2];
          pSrc[0] = y_plane;  // Y plane pointer
          pSrc[1] = uv_plane;
          size_t pitch;
          // rgb_frame_ptr is on device
          cudaMallocPitch(&rgb_frame_ptr, &pitch, framePitch * 3, frameHeight);
          
          stat = nppiNV12ToRGB_8u_P2C3R(pSrc,
                            framePitch, 
                            static_cast<Npp8u*>(rgb_frame_ptr), 
                            pitch,
                            NppiSize {frameWidth, frameHeight});
          if (stat != NPP_SUCCESS) {
            printf("nppiNV12ToRGB_8u_P2C3R failed with error %d", stat);
          }

          cudaMallocHost(&host_ptr, pitch * frameHeight);
          cudaMemcpy((void *)host_ptr,
                      (void *)rgb_frame_ptr,
                      pitch * frameHeight,
                      cudaMemcpyDeviceToHost);

          cudaFree(rgb_frame_ptr);
          info->dataSize = pitch * frameHeight;
          info->width = frameWidth;
          info->height = frameHeight;
          info->colorFormat = colorFormat;
          info->pitch = pitch;
          
        }
        

        
        g_mutex_lock (&dsclipper->data_lock);
        g_queue_push_tail (dsclipper->data_queue, host_ptr);
        g_queue_push_tail (dsclipper->data_queue, info);
        g_mutex_unlock (&dsclipper->data_lock);
        break;
      }
    }
  }
  
  nvtxDomainRangeEnd(dsclipper->nvtx_domain, buf_process_range);

  /* Queue a push buffer batch. This batch is not inferred. This batch is to
   * signal the process thread that there are no more batches
   * belonging to this input buffer and this GstBuffer can be pushed to
   * downstream element once all the previous processing is done. */
  buf_push_batch = new GstDsClipperBatch;
  buf_push_batch->inbuf = inbuf;
  buf_push_batch->push_buffer = TRUE;
  buf_push_batch->nvtx_complete_buf_range = buf_process_range;

  g_mutex_lock (&dsclipper->process_lock);
  /* Check if this is a push buffer or event marker batch. If yes, no need to
   * queue the input for inferencing. */
  if (buf_push_batch->push_buffer) {
    /* Push the batch info structure in the processing queue and notify the
     * process thread that a new batch has been queued. */
    g_queue_push_tail (dsclipper->process_queue, buf_push_batch);
    g_cond_broadcast (&dsclipper->process_cond);
  }
  g_mutex_unlock (&dsclipper->process_lock);

  flow_ret = GST_FLOW_OK;

error:
  gst_buffer_unmap (inbuf, &in_map_info);
  return flow_ret;
}

/**
 * If submit_input_buffer is implemented, it is mandatory to implement
 * generate_output. Buffers are not pushed to the downstream element from here.
 * Return the GstFlowReturn value of the latest pad push so that any error might
 * be caught by the application.
 */
static GstFlowReturn
gst_dsclipper_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf)
{
  GstDsClipper *dsclipper = GST_DSCLIPPER (btrans);
  return dsclipper->last_flow_ret;
}

/**
 * Attach metadata for the full frame. We will be adding a new metadata.
 */
static void
attach_metadata_full_frame (GstDsClipper * dsclipper,
    NvDsFrameMeta * frame_meta, gdouble scale_ratio, DsClipperOutput * output,
    guint batch_id)
{
  NvDsBatchMeta *batch_meta = frame_meta->base_meta.batch_meta;
  NvDsObjectMeta *object_meta = NULL;
  static gchar font_name[] = "Serif";
  GST_DEBUG_OBJECT (dsclipper, "Attaching metadata %d\n", output->numObjects);

  for (gint i = 0; i < output->numObjects; i++) {
    DsClipperObject *obj = &output->object[i];
    object_meta = nvds_acquire_obj_meta_from_pool (batch_meta);
    NvOSD_RectParams & rect_params = object_meta->rect_params;
    NvOSD_TextParams & text_params = object_meta->text_params;

    // Assign bounding box coordinates
    rect_params.left = obj->left;
    rect_params.top = obj->top;
    rect_params.width = obj->width;
    rect_params.height = obj->height;

    // Semi-transparent yellow background
    rect_params.has_bg_color = 0;
    rect_params.bg_color = (NvOSD_ColorParams) {
    1, 1, 0, 0.4};
    // Red border of width 6
    rect_params.border_width = 3;
    rect_params.border_color = (NvOSD_ColorParams) {
    1, 0, 0, 1};

    // Scale the bounding boxes proportionally based on how the object/frame was
    // scaled during input
    rect_params.left /= scale_ratio;
    rect_params.top /= scale_ratio;
    rect_params.width /= scale_ratio;
    rect_params.height /= scale_ratio;
    GST_DEBUG_OBJECT (dsclipper, "Attaching rect%d of batch%u"
        "  left->%f top->%f width->%f"
        " height->%f label->%s\n", i, batch_id, rect_params.left,
        rect_params.top, rect_params.width, rect_params.height, obj->label);

    object_meta->object_id = UNTRACKED_OBJECT_ID;
    g_strlcpy (object_meta->obj_label, obj->label, MAX_LABEL_SIZE);
    // display_text required heap allocated memory
    text_params.display_text = g_strdup (obj->label);
    // Display text above the left top corner of the object
    text_params.x_offset = rect_params.left;
    text_params.y_offset = rect_params.top - 10;
    // Set black background for the text
    text_params.set_bg_clr = 1;
    text_params.text_bg_clr = (NvOSD_ColorParams) {
    0, 0, 0, 1};
    // Font face, size and color
    text_params.font_params.font_name = font_name;
    text_params.font_params.font_size = 11;
    text_params.font_params.font_color = (NvOSD_ColorParams) {
    1, 1, 1, 1};

    nvds_add_obj_meta_to_frame (frame_meta, object_meta, NULL);
  }
}

/**
 * Only update string label in an existing object metadata. No bounding boxes.
 * We assume only one label per object is generated
 */
static void
attach_metadata_object (GstDsClipper * dsclipper, NvDsObjectMeta * obj_meta,
    DsClipperOutput * output)
{
  if (output->numObjects == 0)
    return;
  NvDsBatchMeta *batch_meta = obj_meta->base_meta.batch_meta;

  NvDsClassifierMeta *classifier_meta =
      nvds_acquire_classifier_meta_from_pool (batch_meta);

  classifier_meta->unique_component_id = dsclipper->unique_id;

  NvDsLabelInfo *label_info =
      nvds_acquire_label_info_meta_from_pool (batch_meta);
  g_strlcpy (label_info->result_label, output->object[0].label, MAX_LABEL_SIZE);
  nvds_add_label_info_meta_to_classifier (classifier_meta, label_info);
  nvds_add_classifier_meta_to_object (obj_meta, classifier_meta);

  nvds_acquire_meta_lock (batch_meta);
  NvOSD_TextParams & text_params = obj_meta->text_params;
  NvOSD_RectParams & rect_params = obj_meta->rect_params;

  /* Below code to display the result */
  // Set black background for the text
  // display_text required heap allocated memory
  if (text_params.display_text) {
    gchar *conc_string = g_strconcat (text_params.display_text, " ",
        output->object[0].label, NULL);
    g_free (text_params.display_text);
    text_params.display_text = conc_string;
  } else {
    // Display text above the left top corner of the object
    text_params.x_offset = rect_params.left;
    text_params.y_offset = rect_params.top - 10;
    text_params.display_text = g_strdup (output->object[0].label);
    // Font face, size and color
    text_params.font_params.font_name = (char *) "Serif";
    text_params.font_params.font_size = 11;
    text_params.font_params.font_color = (NvOSD_ColorParams) {
    1, 1, 1, 1};
    // Set black background for the text
    text_params.set_bg_clr = 1;
    text_params.text_bg_clr = (NvOSD_ColorParams) {
    0, 0, 0, 1};
  }
  nvds_release_meta_lock (batch_meta);
}

/**
 * Output loop used to pop output from processing thread, attach the output to the
 * buffer in form of NvDsMeta and push the buffer to downstream element.
 */
static gpointer
gst_dsclipper_output_loop (gpointer data)
{
  GstDsClipper *dsclipper = GST_DSCLIPPER (data);
  DsClipperOutput *output;
  NvDsObjectMeta *obj_meta = NULL;
  gdouble scale_ratio = 1.0;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  std::string nvtx_str;

  nvtx_str =
      "gst-dsclipper_output-loop_uid=" + std::to_string (dsclipper->unique_id);

  g_mutex_lock (&dsclipper->process_lock);

  /* Run till signalled to stop. */
  while (!dsclipper->stop) {
    std::unique_ptr < GstDsClipperBatch > batch = nullptr;

    /* Wait if processing queue is empty. */
    if (g_queue_is_empty (dsclipper->process_queue)) {
      g_cond_wait (&dsclipper->process_cond, &dsclipper->process_lock);
      continue;
    }

    /* Pop a batch from the element's process queue. */
    batch.reset ((GstDsClipperBatch *)
        g_queue_pop_head (dsclipper->process_queue));
    g_cond_broadcast (&dsclipper->process_cond);

    /* Event marker used for synchronization. No need to process further. */
    if (batch->event_marker) {
      continue;
    }

    g_mutex_unlock (&dsclipper->process_lock);

    /* Need to only push buffer to downstream element. This batch was not
     * actually submitted for inferencing. */
    if (batch->push_buffer) {
      nvtxDomainRangeEnd(dsclipper->nvtx_domain, batch->nvtx_complete_buf_range);

      nvds_set_output_system_timestamp (batch->inbuf,
          GST_ELEMENT_NAME (dsclipper));

      GstFlowReturn flow_ret =
          gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (dsclipper),
          batch->inbuf);
      if (dsclipper->last_flow_ret != flow_ret) {
        switch (flow_ret) {
            /* Signal the application for pad push errors by posting a error message
             * on the pipeline bus. */
          case GST_FLOW_ERROR:
          case GST_FLOW_NOT_LINKED:
          case GST_FLOW_NOT_NEGOTIATED:
            GST_ELEMENT_ERROR (dsclipper, STREAM, FAILED,
                ("Internal data stream error."),
                ("streaming stopped, reason %s (%d)",
                    gst_flow_get_name (flow_ret), flow_ret));
            break;
          default:
            break;
        }
      }
      dsclipper->last_flow_ret = flow_ret;
      g_mutex_lock (&dsclipper->process_lock);
      continue;
    }

    nvtx_str = "dequeueOutputAndAttachMeta batch_num=" + std::to_string(batch->inbuf_batch_num);
    eventAttrib.message.ascii = nvtx_str.c_str();
    nvtxDomainRangePushEx(dsclipper->nvtx_domain, &eventAttrib);

    g_mutex_lock (&dsclipper->process_lock);

#ifdef WITH_OPENCV
    g_queue_push_tail (dsclipper->buf_queue, batch->cvmat);
#else
    g_queue_push_tail (dsclipper->buf_queue, batch->inter_buf);
#endif
    g_cond_broadcast (&dsclipper->buf_cond);

    nvtxDomainRangePop (dsclipper->nvtx_domain);
  }
  g_mutex_unlock (&dsclipper->process_lock);

  return nullptr;
}


void saveImageToRaw(const char* filename, void* data, size_t dataSize, int width, int height, int pitch) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Unable to open file for writing: " << filename << std::endl;
        return;
    }

    // Assuming the color format is one of the standard CUDA channel formats
    // For simplicity, let's assume it's unsigned char and 4 channels (e.g., RGBA)
    for (int y = 0; y < height; ++y) {
        file.write(static_cast<const char*>(data) + y * pitch, width * 4); // Assuming 4 bytes per pixel (RGBA)
    }


    file.close();
}

static gpointer
gst_dsclipper_data_loop (gpointer data)
{
  GstDsClipper *dsclipper = GST_DSCLIPPER (data);
  // DsClipperOutput *output;
  NvDsObjectMeta *obj_meta = NULL;
  gdouble scale_ratio = 1.0;
  auto start_time = std::chrono::system_clock::now();
  auto end_time = std::chrono::system_clock::now();
  auto duration = end_time - start_time;
  double duration_seconds;
  gpointer host_ptr = NULL;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  std::string nvtx_str;

  nvtx_str =
      "gst-dsclipper_output-loop_uid=" + std::to_string (dsclipper->unique_id);


  /* Run till signalled to stop. */
  while (!dsclipper->stop) {
    // sleep 1 second
    g_usleep(1000);
    if (g_queue_is_empty (dsclipper->data_queue)) {
      continue;
    }
    
    g_mutex_lock (&dsclipper->data_lock);
    
    host_ptr = g_queue_pop_head(dsclipper->data_queue);
    HostFrameInfo *info = (HostFrameInfo *) g_queue_pop_head (dsclipper->data_queue);
    g_mutex_unlock (&dsclipper->data_lock);

    printf("pop info: %d %d %ld %ld\n", info->width, info->height, info->pitch, info->dataSize);

    // start_time = std::chrono::system_clock::now();
    char image_name[100];
    static guint cnt = 0;
    sprintf(image_name, "out_%d.png", cnt);  
    stbi_write_png(image_name, info->width, info->height, 3, static_cast<unsigned char*>(host_ptr), info->pitch);
    
    // saveImageToRaw(image_name, host_ptr, tmp.dataSize, tmp.width, tmp.height, tmp.pitch);
    // // }
    // end_time = std::chrono::system_clock::now();
    // duration = end_time - start_time;
    // duration_seconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    // printf("time usage of saving: %f\n", duration_seconds);
    cnt++;

    if (host_ptr){
      // printf("release host ptr\n");
      cudaFreeHost(host_ptr);
    }
    g_free(info);
    


    // nvtx_str = "dequeueOutputAndAttachMeta batch_num=" + std::to_string(batch->inbuf_batch_num);
    // eventAttrib.message.ascii = nvtx_str.c_str();
    // nvtxDomainRangePushEx(dsclipper->nvtx_domain, &eventAttrib);

    // static guint cnt = 0;
    // void *host_ptr = NULL;
    // printf("frame size in this batch: %ld\n", batch->frames.size ());
    // /* For each frame attach metadata output. */
    // for (guint i = 0; i < batch->frames.size (); i++) {
    //   NvBufSurfaceParams tmp = batch->inter_buf->surfaceList[i];
    //   // printf("info: width %d, height %d, pitch %d, size %d\n", tmp.width, tmp.height, tmp.pitch, tmp.dataSize);
      
      
    //   cudaMallocHost(&host_ptr, tmp.dataSize);
    //   start_time = std::chrono::system_clock::now();
    //   cudaMemcpy((void *)host_ptr,
    //             (void *)tmp.dataPtr,
    //             tmp.dataSize,
    //             cudaMemcpyDeviceToHost);
    //   end_time = std::chrono::system_clock::now();
    //   duration = end_time - start_time;
    //   duration_seconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    //   // printf("time usage of memcpy: %f\n", duration_seconds);
    //   start_time = std::chrono::system_clock::now();
    //   char image_name[100];
    //   sprintf(image_name, "out_%d.raw", cnt);  
    //   // stbi_write_png(image_name, tmp.width, tmp.height, 4, static_cast<unsigned char*>(host_ptr), tmp.pitch);
      
    //   saveImageToRaw(image_name, host_ptr, tmp.dataSize, tmp.width, tmp.height, tmp.pitch);
    //   // }
    //   end_time = std::chrono::system_clock::now();
    //   duration = end_time - start_time;
    //   duration_seconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    //   printf("time usage of saving: %f\n", duration_seconds);
    //   cnt++;
    //   cudaFreeHost(host_ptr);
      
    // }
    
//     g_mutex_lock (&dsclipper->data_lock);

// #ifdef WITH_OPENCV
//     g_queue_push_tail (dsclipper->buf_queue, batch->cvmat);
// #else
//     g_queue_push_tail (dsclipper->buf_queue, batch->inter_buf);
// #endif
//     g_cond_broadcast (&dsclipper->buf_cond);

//     nvtxDomainRangePop (dsclipper->nvtx_domain);
//     printf("data done\n");
  }

  return nullptr;
}



/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
dsclipper_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_dsclipper_debug, "dsclipper", 0,
      "dsclipper plugin");

  return gst_element_register (plugin, "dsclipper", GST_RANK_PRIMARY,
      GST_TYPE_DSCLIPPER);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_dsclipper,
    DESCRIPTION, dsclipper_plugin_init, "6.3", LICENSE, BINARY_PACKAGE,
    URL)

