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
  PROP_GPU_DEVICE_ID,
  PROP_OPERATE_ON_GIE_ID,
  PROP_OPERATE_ON_CLASS_IDS,
  PROP_NAME_FORMAT,
  PROP_OUTPUT_PATH
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
#define DEFAULT_GPU_ID 0
#define DEFAULT_OPERATE_ON_GIE_ID -1
#define DEFAULT_OUTPUT_PATH ""
#define DEFAULT_NAME_FORMAT "frame;trackid;classid;conf"
#define DEFAULT_INTERVAL 0

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

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id",
          "Set GPU Device ID",
          "Set GPU Device ID", 0,
          G_MAXUINT, 0,
          GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OUTPUT_PATH,
      g_param_spec_string ("output-path", "Output Path",
          "Path to save images for this instance of dsclipper",
          DEFAULT_OUTPUT_PATH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_PLAYING)));
              
  g_object_class_install_property (gobject_class, PROP_NAME_FORMAT,
      g_param_spec_string ("name-format", "File Name Format",
          "Format of the output file name.\n",
          "\t\t\t frameidx_trackid_classid_conf."
          DEFAULT_NAME_FORMAT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_PLAYING)));


  g_object_class_install_property (gobject_class, PROP_OPERATE_ON_GIE_ID,
      g_param_spec_int ("infer-on-gie-id", "Infer on Gie ID",
          "Infer on metadata generated by GIE with this unique ID.\n"
          "\t\t\tSet to -1 to infer on all metadata.",
          -1, G_MAXINT, DEFAULT_OPERATE_ON_GIE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OPERATE_ON_CLASS_IDS,
      g_param_spec_string ("operated-on-class-ids", "Operate on Class ids",
          "Operate on objects with specified class ids\n"
          "\t\t\tUse string with values of class ids in ClassID (int) to set the property.\n"
          "\t\t\t e.g. 0:2:3",
          "",
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));
  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsclipper_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsclipper_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "DsClipper plugin",
      "DsClipper Plugin",
      "Clip objects based on Deepstream nvinfer outputs",
      "DAMON ZHOU "
      "@ https://github.com/zhouyuchong/gst-dsclipper");
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
  dsclipper->gpu_id = DEFAULT_GPU_ID;
  dsclipper->operate_on_gie_id = DEFAULT_OPERATE_ON_GIE_ID;
  dsclipper->operate_on_class_ids = new std::vector < gboolean >;
  dsclipper->output_path = g_strdup (DEFAULT_OUTPUT_PATH);
  dsclipper->name_format = g_strdup (DEFAULT_NAME_FORMAT);
  dsclipper->interval = DEFAULT_INTERVAL;
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
    case PROP_GPU_DEVICE_ID:
      dsclipper->gpu_id = g_value_get_uint (value);
      break;
    case PROP_NAME_FORMAT:
      dsclipper->name_format = g_value_dup_string (value);
      break;
    case PROP_OUTPUT_PATH:
      dsclipper->output_path = g_value_dup_string (value);
      break;

    case PROP_OPERATE_ON_GIE_ID:
      dsclipper->operate_on_gie_id = g_value_get_int (value);
      break;
      
    case PROP_OPERATE_ON_CLASS_IDS:
    {
      std::stringstream str (g_value_get_string (value));
      std::vector < gint > class_ids;
      gint max_class_id = -1;

      while (str.peek () != EOF) {
        gint class_id;
        str >> class_id;
        class_ids.push_back (class_id);
        max_class_id = MAX (max_class_id, class_id);
        str.get ();
      }
      dsclipper->operate_on_class_ids->assign (max_class_id + 1, FALSE);
    for (auto & cid:class_ids)
        dsclipper->operate_on_class_ids->at (cid) = TRUE;
    }
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
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, dsclipper->gpu_id);
      break;
    case PROP_OPERATE_ON_GIE_ID:
      g_value_set_int (value, dsclipper->operate_on_gie_id);
      break;
    case PROP_OPERATE_ON_CLASS_IDS:
    {
      std::stringstream str;
      for (size_t i = 0; i < dsclipper->operate_on_class_ids->size (); i++) {
        if (dsclipper->operate_on_class_ids->at (i))
          str << i << ":";
      }
      g_value_set_string (value, str.str ().c_str ());
    }
      break;

    case PROP_OUTPUT_PATH:
      dsclipper->output_path = g_value_dup_string (value);
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

  nvtx_str = "GstNvDsClipper: UID=" + std::to_string(dsclipper->unique_id);
  auto nvtx_deleter = [](nvtxDomainHandle_t d) { nvtxDomainDestroy (d); };
  std::unique_ptr<nvtxDomainRegistration, decltype(nvtx_deleter)> nvtx_domain_ptr (
      nvtxDomainCreate(nvtx_str.c_str()), nvtx_deleter);

  CHECK_CUDA_STATUS (cudaSetDevice (dsclipper->gpu_id),
      "Unable to set cuda device");

  CHECK_CUDA_STATUS (cudaStreamCreate (&dsclipper->cuda_stream),
      "Could not create cuda stream");

  /* Create process queue and cvmat queue to transfer data between threads.
   * We will be using this queue to maintain the list of frames/objects
   * currently given to the algorithm for processing. */
  dsclipper->process_queue = g_queue_new ();
  dsclipper->buf_queue = g_queue_new ();
  dsclipper->data_queue = g_queue_new ();


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

  if (dsclipper->cuda_stream) {
    cudaStreamDestroy (dsclipper->cuda_stream);
    dsclipper->cuda_stream = NULL;
  }
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

static inline gboolean
should_crop_object (GstDsClipper *dsclipper, NvDsObjectMeta * obj_meta, gulong frame_num)
{
  if (dsclipper->operate_on_gie_id > -1 &&
      obj_meta->unique_component_id != dsclipper->operate_on_gie_id)
    return FALSE;

  if (!dsclipper->operate_on_class_ids->empty () &&
      ((int) dsclipper->operate_on_class_ids->size () <= obj_meta->class_id ||
          dsclipper->operate_on_class_ids->at (obj_meta->class_id) == FALSE)) {
    return FALSE;
  }
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

  gboolean need_clip = FALSE;  
  

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    frame_meta = (NvDsFrameMeta *) (l_frame->data);
    void *host_ptr = NULL;
    void *rgb_frame_ptr = NULL;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);

      need_clip = should_crop_object (dsclipper, obj_meta, frame_meta->frame_num);
      // printf("cls id: %d, need clip: %d\n", obj_meta->class_id, need_clip);
      // std::cout<<obj_meta->object_id<<" "<<UNTRACKED_OBJECT_ID<<std::endl;
      // continue;
      if (!need_clip) continue;

      // /* Should not process on objects smaller than MIN_INPUT_OBJECT_WIDTH x MIN_INPUT_OBJECT_HEIGHT
      //   * since it will cause hardware scaling issues. */
      // if (obj_meta->rect_params.width < MIN_INPUT_OBJECT_WIDTH ||
      //     obj_meta->rect_params.height < MIN_INPUT_OBJECT_HEIGHT)
      //   continue;

      // if (!nvinfer->operate_on_class_ids->empty () &&
      //     ((int) nvinfer->operate_on_class_ids->size () <= obj_meta->class_id ||
      //         nvinfer->operate_on_class_ids->at (obj_meta->class_id) == FALSE)) {
      // }
      

      if (g_queue_get_length(dsclipper->data_queue) < MAX_QUEUE_SIZE) {
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
 * Output loop used to pop output from processing thread, attach the output to the
 * buffer in form of NvDsMeta and push the buffer to downstream element.
 */
static gpointer
gst_dsclipper_output_loop (gpointer data)
{
  GstDsClipper *dsclipper = GST_DSCLIPPER (data);
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

