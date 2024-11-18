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

#include "gstdscropper.h"

#include <sys/time.h>
#include <condition_variable>
#include <mutex>
#include <thread>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

GST_DEBUG_CATEGORY_STATIC (gst_dscropper_debug);
#define GST_CAT_DEFAULT gst_dscropper_debug
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
  PROP_OUTPUT_PATH,
  PROP_INTERVAL,
  PROP_SCALE_RATIO,
  PROP_CROP_MODE
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
#define DEFAULT_INTERVAL -1
#define DEFAULT_SCALE_RATIO 1.0
#define DEFAULT_CROP_MODE 0

#define RGB_BYTES_PER_PIXEL 3
#define RGBA_BYTES_PER_PIXEL 4
#define Y_BYTES_PER_PIXEL 1
#define UV_BYTES_PER_PIXEL 2

#define MIN_INPUT_OBJECT_WIDTH 16
#define MIN_INPUT_OBJECT_HEIGHT 16

#define MAX_QUEUE_SIZE 20
#define MAX_OBJ_INFO_SIZE 50


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
static GstStaticPadTemplate gst_dscropper_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_dscropper_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_dscropper_parent_class parent_class
G_DEFINE_TYPE (GstDsCropper, gst_dscropper, GST_TYPE_BASE_TRANSFORM);

static void gst_dscropper_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_dscropper_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_dscropper_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_dscropper_start (GstBaseTransform * btrans);
static gboolean gst_dscropper_stop (GstBaseTransform * btrans);

static GstFlowReturn
gst_dscropper_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf);
static GstFlowReturn
gst_dscropper_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf);


static gpointer gst_dscropper_output_loop (gpointer data);

static gpointer gst_dscropper_data_loop (gpointer data);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_dscropper_class_init (GstDsCropperClass * klass)
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
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_dscropper_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_dscropper_get_property);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_dscropper_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_dscropper_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_dscropper_stop);

  gstbasetransform_class->submit_input_buffer =
      GST_DEBUG_FUNCPTR (gst_dscropper_submit_input_buffer);
  gstbasetransform_class->generate_output =
      GST_DEBUG_FUNCPTR (gst_dscropper_generate_output);

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

  g_object_class_install_property (gobject_class, PROP_CROP_MODE,
      g_param_spec_uint ("crop-mode",
          "crop mode",
          "Set crop mode, 0 defalt only crop object, 1 crop both object and original frame", 0,
          G_MAXUINT, 0,
          GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));
              
  g_object_class_install_property (gobject_class, PROP_SCALE_RATIO,
      g_param_spec_float ("scale-ratio",
          "Crop Scale Ratio",
          "Crop scale Ratio", 1.0,
          G_MAXFLOAT, 1.0,
          GParamFlags
          (G_PARAM_READWRITE |
              G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_OUTPUT_PATH,
      g_param_spec_string ("output-path", "Output Path",
          "Path to save images for this instance of dscropper",
          DEFAULT_OUTPUT_PATH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_PLAYING)));
              
  g_object_class_install_property (gobject_class, PROP_NAME_FORMAT,
      g_param_spec_string ("name-format", "File Name Format",
          "Format of the output file name.\n"
          "\t\t\t frameidx_trackid_classid_conf.",
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

  g_object_class_install_property (gobject_class, PROP_INTERVAL,
      g_param_spec_int ("interval", "Interval",
          "Specifies number of consecutive batches to be skipped for clip",
          -1, G_MAXINT, DEFAULT_INTERVAL,
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
      gst_static_pad_template_get (&gst_dscropper_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dscropper_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class,
      "DsCropper plugin",
      "DsCropper Plugin",
      "Clip objects based on Deepstream nvinfer outputs",
      "DAMON ZHOU "
      "@ https://github.com/zhouyuchong/gst-dscropper");
}

static void
gst_dscropper_init (GstDsCropper * dscropper)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (dscropper);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  /* Initialize all property variables to default values */
  dscropper->unique_id = DEFAULT_UNIQUE_ID;
  dscropper->gpu_id = DEFAULT_GPU_ID;
  dscropper->operate_on_gie_id = DEFAULT_OPERATE_ON_GIE_ID;
  dscropper->operate_on_class_ids = new std::vector < gboolean >;
  dscropper->output_path = g_strdup (DEFAULT_OUTPUT_PATH);
  dscropper->name_format = g_strdup (DEFAULT_NAME_FORMAT);
  dscropper->interval = DEFAULT_INTERVAL;
  dscropper->scale_ratio = DEFAULT_SCALE_RATIO;
  dscropper->crop_mode = DEFAULT_CROP_MODE;
  /* This quark is required to identify NvDsMeta when iterating through
   * the buffer metadatas */
  if (!_dsmeta_quark)
    _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_dscropper_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsCropper *dscropper = GST_DSCROPPER (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      dscropper->unique_id = g_value_get_uint (value);
      break;
    case PROP_GPU_DEVICE_ID:
      dscropper->gpu_id = g_value_get_uint (value);
      break;
    case PROP_NAME_FORMAT:
      dscropper->name_format = g_value_dup_string (value);
      break;
    case PROP_OUTPUT_PATH:
      dscropper->output_path = g_value_dup_string (value);
      break;
    case PROP_SCALE_RATIO:
      dscropper->scale_ratio = g_value_get_float (value);
      break;
    case PROP_CROP_MODE:
      dscropper->crop_mode = g_value_get_uint (value);
      break;
    case PROP_OPERATE_ON_GIE_ID:
      dscropper->operate_on_gie_id = g_value_get_int (value);
      break;
      
    case PROP_INTERVAL:
      dscropper->interval = g_value_get_int (value);
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
      dscropper->operate_on_class_ids->assign (max_class_id + 1, FALSE);
    for (auto & cid:class_ids)
        dscropper->operate_on_class_ids->at (cid) = TRUE;
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
gst_dscropper_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsCropper *dscropper = GST_DSCROPPER (object);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, dscropper->unique_id);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, dscropper->gpu_id);
      break;
    case PROP_OPERATE_ON_GIE_ID:
      g_value_set_int (value, dscropper->operate_on_gie_id);
      break;
    case PROP_OPERATE_ON_CLASS_IDS:
    {
      std::stringstream str;
      for (size_t i = 0; i < dscropper->operate_on_class_ids->size (); i++) {
        if (dscropper->operate_on_class_ids->at (i))
          str << i << ":";
      }
      g_value_set_string (value, str.str ().c_str ());
    }
      break;

    case PROP_INTERVAL:
      g_value_set_int (value, dscropper->interval);
      break;
    case PROP_CROP_MODE:
      g_value_set_uint (value, dscropper->crop_mode);
      break;
    case PROP_SCALE_RATIO:
      g_value_set_float (value, dscropper->scale_ratio);
      break;

    case PROP_OUTPUT_PATH:
      dscropper->output_path = g_value_dup_string (value);
      break;

    case PROP_NAME_FORMAT:
      dscropper->name_format = g_value_dup_string (value);
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
gst_dscropper_start (GstBaseTransform * btrans)
{
  GstDsCropper *dscropper = GST_DSCROPPER (btrans);
  std::string nvtx_str;
#ifdef WITH_OPENCV
  // OpenCV mat containing RGB data
  cv::Mat * cvmat;
#else
  NvBufSurface * inter_buf;
#endif

  nvtx_str = "GstNvDsCropper: UID=" + std::to_string(dscropper->unique_id);
  auto nvtx_deleter = [](nvtxDomainHandle_t d) { nvtxDomainDestroy (d); };
  std::unique_ptr<nvtxDomainRegistration, decltype(nvtx_deleter)> nvtx_domain_ptr (
      nvtxDomainCreate(nvtx_str.c_str()), nvtx_deleter);

  CHECK_CUDA_STATUS (cudaSetDevice (dscropper->gpu_id),
      "Unable to set cuda device");

  CHECK_CUDA_STATUS (cudaStreamCreate (&dscropper->cuda_stream),
      "Could not create cuda stream");

  /* Create process queue and cvmat queue to transfer data between threads.
   * We will be using this queue to maintain the list of frames/objects
   * currently given to the algorithm for processing. */
  dscropper->process_queue = g_queue_new ();
  dscropper->buf_queue = g_queue_new ();
  dscropper->data_queue = g_queue_new ();
  dscropper->object_infos = new std::unordered_map<guint64, CropperObjectInfo>();
  dscropper->insertion_order = new std::list<guint64>();

  /* Start a thread which will pop output from the algorithm, form NvDsMeta and
   * push buffers to the next element. */
  dscropper->process_thread =
      g_thread_new ("dscropper-process-thread", gst_dscropper_output_loop,
      dscropper);
  dscropper->data_thread =
      g_thread_new ("dscropper-data-thread", gst_dscropper_data_loop,
      dscropper);
  dscropper->nvtx_domain = nvtx_domain_ptr.release ();

  return TRUE;
error:

  if (dscropper->cuda_stream) {
    cudaStreamDestroy (dscropper->cuda_stream);
    dscropper->cuda_stream = NULL;
  }
  return FALSE;
}

/**
 * Stop the process thread and free up all the resources
 */
static gboolean
gst_dscropper_stop (GstBaseTransform * btrans)
{
  GstDsCropper *dscropper = GST_DSCROPPER (btrans);

#ifdef WITH_OPENCV
  cv::Mat * cvmat;
#else
  NvBufSurface * inter_buf;
#endif

  g_mutex_lock (&dscropper->process_lock);

  /* Wait till all the items in the queue are handled. */
  while (!g_queue_is_empty (dscropper->process_queue)) {
    g_cond_wait (&dscropper->process_cond, &dscropper->process_lock);
  }

  g_mutex_lock (&dscropper->data_lock);
  while (!g_queue_is_empty (dscropper->data_queue)) {
    g_cond_wait (&dscropper->data_cond, &dscropper->data_lock);
  }

#ifdef WITH_OPENCV
  while (!g_queue_is_empty (dscropper->buf_queue)) {
    cvmat = (cv::Mat *) g_queue_pop_head (dscropper->buf_queue);
    delete[]cvmat;
    cvmat = NULL;
  }
#else
  while (!g_queue_is_empty (dscropper->buf_queue)) {
    inter_buf = (NvBufSurface *) g_queue_pop_head (dscropper->buf_queue);
    if (inter_buf)
      NvBufSurfaceDestroy (inter_buf);
    inter_buf = NULL;
  }
#endif
  dscropper->stop = TRUE;

  g_cond_broadcast (&dscropper->process_cond);
  g_mutex_unlock (&dscropper->process_lock);

  g_cond_broadcast (&dscropper->data_cond);
  g_mutex_unlock (&dscropper->data_lock);

  g_thread_join (dscropper->process_thread);
  g_thread_join (dscropper->data_thread);

#ifdef WITH_OPENCV
  if (dscropper->inter_buf)
    NvBufSurfaceDestroy (dscropper->inter_buf);
  dscropper->inter_buf = NULL;
#endif

  if (dscropper->cuda_stream)
    cudaStreamDestroy (dscropper->cuda_stream);
  dscropper->cuda_stream = NULL;

  if (dscropper->object_infos) delete dscropper->object_infos;
  dscropper->object_infos = nullptr;
  delete dscropper->insertion_order;
  dscropper->insertion_order = nullptr;

  g_queue_free (dscropper->process_queue);
  g_queue_free (dscropper->data_queue);
  g_queue_free (dscropper->buf_queue);
  
  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_dscropper_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstDsCropper *dscropper = GST_DSCROPPER (btrans);
  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps (&dscropper->video_info, incaps);

  CHECK_CUDA_STATUS (cudaSetDevice (dscropper->gpu_id),
      "Unable to set cuda device");

  return TRUE;

error:
  return FALSE;
}

std::string formatString(const char* input, int frame_num, int track_id, int class_id, float conf) {
    std::istringstream iss(input);
    std::string token;
    std::ostringstream oss; 

    while (getline(iss, token, ';')) {
        if (token == "frame") {
            oss << "frm" << frame_num << "_";
        } else if (token == "trackid") {
            oss << "tid" << track_id << "_";
        } else if (token == "classid") {
            oss << "cls" << class_id << "_";
        } else if (token == "conf") {
            oss << "conf" << (int)(conf*100) << "_";
        } else {
            std::cerr << "Unknown category: " << token << std::endl;
        }
    }

    std::string result = oss.str();
    // 移除末尾的逗号和空格
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
        result.pop_back();
    }

    return result;
}

static inline gboolean
should_crop_object (GstDsCropper *dscropper, NvDsObjectMeta * obj_meta, guint counter)
{
  if (dscropper->operate_on_gie_id > -1 &&
      obj_meta->unique_component_id != dscropper->operate_on_gie_id)
    return FALSE;

  if (!dscropper->operate_on_class_ids->empty () &&
      ((int) dscropper->operate_on_class_ids->size () <= obj_meta->class_id ||
          dscropper->operate_on_class_ids->at (obj_meta->class_id) == FALSE)) {
    return FALSE;
  }
  if (obj_meta->object_id  == UNTRACKED_OBJECT_ID) {
    GST_WARNING_OBJECT (dscropper, "Untracked objects in metadata. Cannot"
      " use interval mode on untracked objects.");
    return FALSE;
  }
  if (dscropper->interval == -1 && counter) return FALSE;
  if (dscropper->interval > 0) {
    if (counter % (dscropper->interval + 1) != 0) return FALSE;
  }
  return TRUE;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_dscropper_submit_input_buffer (GstBaseTransform * btrans,
    gboolean discont, GstBuffer * inbuf)
{
  GstDsCropper *dscropper = GST_DSCROPPER (btrans);
  GstMapInfo in_map_info;
  NvBufSurface *in_surf;
  GstDsCropperBatch *buf_push_batch;
  GstFlowReturn flow_ret;
  std::string nvtx_str;
  std::unique_ptr < GstDsCropperBatch > batch = nullptr;

  NvDsBatchMeta *batch_meta = NULL;
  guint i = 0;
  gdouble scale_ratio = 1.0;
  guint num_filled = 0;

  dscropper->current_batch_num++;

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = 0xFFFF0000;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  nvtx_str = "buffer_process batch_num=" + std::to_string(dscropper->current_batch_num);
  eventAttrib.message.ascii = nvtx_str.c_str();
  nvtxRangeId_t buf_process_range = nvtxDomainRangeStartEx(dscropper->nvtx_domain, &eventAttrib);

  memset (&in_map_info, 0, sizeof (in_map_info));

  /* Map the buffer contents and get the pointer to NvBufSurface. */
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    GST_ELEMENT_ERROR (dscropper, STREAM, FAILED,
        ("%s:gst buffer map to get pointer to NvBufSurface failed", __func__), (NULL));
    return GST_FLOW_ERROR;
  }
  in_surf = (NvBufSurface *) in_map_info.data;

  nvds_set_input_system_timestamp (inbuf, GST_ELEMENT_NAME (dscropper));

  batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (batch_meta == nullptr) {
    GST_ELEMENT_ERROR (dscropper, STREAM, FAILED,
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
  char surface_name[200];

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    frame_meta = (NvDsFrameMeta *) (l_frame->data);
    void *frame_ptr_dev = NULL;
    void *frame_ptr_host = NULL;
    void *cropped_ptr_host = NULL;
    
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);

      if (obj_meta->object_id == UNTRACKED_OBJECT_ID) {
        GST_WARNING_OBJECT(dscropper, "Untracked objects in metadata. Cannot apply clipping on untracked objects.");
        printf("untracked\n");
        continue;
      }

      auto it = dscropper->object_infos->find(obj_meta->object_id);
      if (it != dscropper->object_infos->end()) {
          it->second.counter += 1;
          it->second.width = obj_meta->rect_params.width;
          it->second.height = obj_meta->rect_params.height;
          it->second.left = obj_meta->rect_params.left;
          it->second.top = obj_meta->rect_params.top;
      } else {
        if (dscropper->object_infos->size() == MAX_OBJ_INFO_SIZE) {
          dscropper->object_infos->erase(dscropper->insertion_order->front());
          dscropper->insertion_order->pop_front();
        }
          CropperObjectInfo object_info;
          object_info.counter = 0;
          object_info.width = obj_meta->rect_params.width;
          object_info.height = obj_meta->rect_params.height;
          object_info.left = obj_meta->rect_params.left;
          object_info.top = obj_meta->rect_params.top;
          dscropper->object_infos->insert(std::make_pair(obj_meta->object_id, object_info));
          dscropper->insertion_order->push_back(obj_meta->object_id);
      }
      it = dscropper->object_infos->find(obj_meta->object_id);
      need_clip = should_crop_object (dscropper, obj_meta, it->second.counter);

      if (!need_clip) continue;

      if (g_queue_get_length(dscropper->data_queue) >= MAX_QUEUE_SIZE) continue;
      
      NvBufSurfaceParams * currentFrameParams = in_surf->surfaceList + frame_meta->batch_id;
      NvBufSurfaceColorFormat colorFormat = currentFrameParams->colorFormat;
      size_t framePitch = currentFrameParams->pitch;
      size_t frameSize = currentFrameParams->dataSize;
      uint frameWidth = currentFrameParams->width;
      uint frameHeight = currentFrameParams->height;
      size_t pitch = 0;
      uint channel = 3;
      // set channle to 4 if colorformat is rgba like
      if (colorFormat > 18 && colorFormat < 27) channel = 4;
      
      /*
      get Nvsurface and transform them into rgb format
      for rgba, there should be 4 channel and dest-colorformat will be rgba too
      */
      if (colorFormat < 19 || colorFormat > 28) {
        CHECK_CUDA_STATUS (cudaMallocPitch(&frame_ptr_dev, &pitch, framePitch * channel, frameHeight), "Could not allocate mem for RGBA frame.");
        const Npp8u* y_plane;                // Pointer to the Y plane
        const Npp8u* uv_plane;               // Pointer to the UV plane
        y_plane = static_cast<const Npp8u*>(currentFrameParams->dataPtr);
        uv_plane = y_plane + (framePitch * frameHeight);
        const Npp8u* pSrc[2];
        pSrc[0] = y_plane;  // Y plane pointer
        pSrc[1] = uv_plane; // UV plane pointer
        
        // transform NV12 to RGB
        stat = nppiNV12ToRGB_8u_P2C3R(pSrc,
                          framePitch, 
                          static_cast<Npp8u*>(frame_ptr_dev), 
                          pitch,
                          NppiSize {static_cast<int>(frameWidth), static_cast<int>(frameHeight)});
        if (stat != NPP_SUCCESS) {
          printf("nppiNV12ToRGB_8u_P2C3R failed with error %d\n", stat);
        }
      } else {
        CHECK_CUDA_STATUS (cudaMalloc(&frame_ptr_dev, frameSize),
                            "Could not allocate mem for RGB frame.");
        cudaMemcpy((void *)frame_ptr_dev,
                (void *)currentFrameParams->dataPtr,
                frameSize,
                cudaMemcpyDeviceToDevice);
        pitch = framePitch * channel;
        // TODO assert frameSize equal to pitch * height
      }
      
      int calculated_width = static_cast<int>(ceil(obj_meta->rect_params.width * dscropper->scale_ratio));
      int calculated_height = static_cast<int>(ceil(obj_meta->rect_params.height * dscropper->scale_ratio));
      int calculated_left = static_cast<int>(obj_meta->rect_params.left - (dscropper->scale_ratio - 1) * obj_meta->rect_params.width / 2);
      int calculated_top = static_cast<int>(obj_meta->rect_params.top - (dscropper->scale_ratio - 1) * obj_meta->rect_params.height / 2);
      
      ClippedSurfaceInfo* info = g_new(ClippedSurfaceInfo, 1);
      memset(surface_name, 0, sizeof(surface_name));
      std::string formattedString = formatString(dscropper->name_format, frame_meta->frame_num, obj_meta->object_id, obj_meta->class_id, obj_meta->confidence);
      sprintf(surface_name, "%s/%sobj.png", dscropper->output_path, formattedString.c_str());  
      info->filename = g_strdup(surface_name);
      info->width = calculated_width;
      info->height = calculated_height;
      info->pitch = calculated_width * channel;
      info->channel = channel;
      
      CHECK_CUDA_STATUS(cudaMallocHost(&cropped_ptr_host, calculated_width * calculated_height * channel), "Could not allocate mem for host buffer.");

      if (colorFormat > 18 && colorFormat < 27) {
        stat = nppiResize_8u_C4R(static_cast<const Npp8u*>(frame_ptr_dev), 
                          pitch, 
                          NppiSize {static_cast<int>(frameWidth), static_cast<int>(frameHeight)}, 
                          NppiRect {calculated_left, calculated_top, calculated_width, calculated_height}, 
                          static_cast<Npp8u*>(cropped_ptr_host), 
                          calculated_width * channel, 
                          NppiSize {calculated_width, calculated_height},
                          NppiRect {0, 0, calculated_width, calculated_height},
                          NPPI_INTER_LINEAR);
      } else {
        stat = nppiResize_8u_C3R(static_cast<const Npp8u*>(frame_ptr_dev), 
                          pitch, 
                          NppiSize {static_cast<int>(frameWidth), static_cast<int>(frameHeight)}, 
                          NppiRect {calculated_left, calculated_top, calculated_width, calculated_height}, 
                          static_cast<Npp8u*>(cropped_ptr_host), 
                          calculated_width * channel, 
                          NppiSize {calculated_width, calculated_height},
                          NppiRect {0, 0, calculated_width, calculated_height},
                          NPPI_INTER_LINEAR);
      }
      if (stat != NPP_SUCCESS) {
        printf("nppiResize failed with error %d\n", stat);
      }
    
      g_mutex_lock (&dscropper->data_lock);
      g_queue_push_tail (dscropper->data_queue, cropped_ptr_host);
      g_queue_push_tail (dscropper->data_queue, info);
      g_mutex_unlock (&dscropper->data_lock);
      
      if (dscropper->crop_mode) {
        CHECK_CUDA_STATUS(cudaMallocHost(&frame_ptr_host, pitch * frameHeight), "Could not allocate mem for host frame buffer.");
        cudaMemcpy((void *)frame_ptr_host,
        (void *)frame_ptr_dev,
        pitch * frameHeight,
        cudaMemcpyDeviceToHost);
        ClippedSurfaceInfo* info_frame = g_new(ClippedSurfaceInfo, 1);
        memset(surface_name, 0, sizeof(surface_name));
        // std::string fs = formatString(dscropper->name_format, frame_meta->frame_num, obj_meta->object_id, obj_meta->class_id, obj_meta->confidence);
        sprintf(surface_name, "%s/%sfrm.png", dscropper->output_path, formattedString.c_str());  
        info_frame->filename = g_strdup(surface_name);
        info_frame->width = frameWidth;
        info_frame->height = frameHeight;
        info_frame->pitch = pitch;
        info_frame->channel = channel;

        g_mutex_lock (&dscropper->data_lock);
        g_queue_push_tail (dscropper->data_queue, frame_ptr_host);
        g_queue_push_tail (dscropper->data_queue, info_frame);
        g_mutex_unlock (&dscropper->data_lock);
        
      }
      // break;
    }
    cudaFree(frame_ptr_dev);
  }
  
  nvtxDomainRangeEnd(dscropper->nvtx_domain, buf_process_range);

  /* Queue a push buffer batch. This batch is not inferred. This batch is to
   * signal the process thread that there are no more batches
   * belonging to this input buffer and this GstBuffer can be pushed to
   * downstream element once all the previous processing is done. */
  buf_push_batch = new GstDsCropperBatch;
  buf_push_batch->inbuf = inbuf;
  buf_push_batch->push_buffer = TRUE;
  buf_push_batch->nvtx_complete_buf_range = buf_process_range;

  g_mutex_lock (&dscropper->process_lock);
  /* Check if this is a push buffer or event marker batch. If yes, no need to
   * queue the input for inferencing. */
  if (buf_push_batch->push_buffer) {
    /* Push the batch info structure in the processing queue and notify the
     * process thread that a new batch has been queued. */
    g_queue_push_tail (dscropper->process_queue, buf_push_batch);
    g_cond_broadcast (&dscropper->process_cond);
  }
  g_mutex_unlock (&dscropper->process_lock);

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
gst_dscropper_generate_output (GstBaseTransform * btrans, GstBuffer ** outbuf)
{
  GstDsCropper *dscropper = GST_DSCROPPER (btrans);
  return dscropper->last_flow_ret;
}

/**
 * Output loop used to pop output from processing thread, attach the output to the
 * buffer in form of NvDsMeta and push the buffer to downstream element.
 */
static gpointer
gst_dscropper_output_loop (gpointer data)
{
  GstDsCropper *dscropper = GST_DSCROPPER (data);
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
      "gst-dscropper_output-loop_uid=" + std::to_string (dscropper->unique_id);

  g_mutex_lock (&dscropper->process_lock);

  /* Run till signalled to stop. */
  while (!dscropper->stop) {
    std::unique_ptr < GstDsCropperBatch > batch = nullptr;

    /* Wait if processing queue is empty. */
    if (g_queue_is_empty (dscropper->process_queue)) {
      g_cond_wait (&dscropper->process_cond, &dscropper->process_lock);
      continue;
    }

    /* Pop a batch from the element's process queue. */
    batch.reset ((GstDsCropperBatch *)
        g_queue_pop_head (dscropper->process_queue));
    g_cond_broadcast (&dscropper->process_cond);

    /* Event marker used for synchronization. No need to process further. */
    if (batch->event_marker) {
      continue;
    }

    g_mutex_unlock (&dscropper->process_lock);

    /* Need to only push buffer to downstream element. This batch was not
     * actually submitted for inferencing. */
    if (batch->push_buffer) {
      nvtxDomainRangeEnd(dscropper->nvtx_domain, batch->nvtx_complete_buf_range);

      nvds_set_output_system_timestamp (batch->inbuf,
          GST_ELEMENT_NAME (dscropper));

      GstFlowReturn flow_ret =
          gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (dscropper),
          batch->inbuf);
      if (dscropper->last_flow_ret != flow_ret) {
        switch (flow_ret) {
            /* Signal the application for pad push errors by posting a error message
             * on the pipeline bus. */
          case GST_FLOW_ERROR:
          case GST_FLOW_NOT_LINKED:
          case GST_FLOW_NOT_NEGOTIATED:
            GST_ELEMENT_ERROR (dscropper, STREAM, FAILED,
                ("Internal data stream error."),
                ("streaming stopped, reason %s (%d)",
                    gst_flow_get_name (flow_ret), flow_ret));
            break;
          default:
            break;
        }
      }
      dscropper->last_flow_ret = flow_ret;
      g_mutex_lock (&dscropper->process_lock);
      continue;
    }

    nvtx_str = "dequeueOutputAndAttachMeta batch_num=" + std::to_string(batch->inbuf_batch_num);
    eventAttrib.message.ascii = nvtx_str.c_str();
    nvtxDomainRangePushEx(dscropper->nvtx_domain, &eventAttrib);

    g_mutex_lock (&dscropper->process_lock);

#ifdef WITH_OPENCV
    g_queue_push_tail (dscropper->buf_queue, batch->cvmat);
#else
    g_queue_push_tail (dscropper->buf_queue, batch->inter_buf);
#endif
    g_cond_broadcast (&dscropper->buf_cond);

    nvtxDomainRangePop (dscropper->nvtx_domain);
  }
  g_mutex_unlock (&dscropper->process_lock);

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
gst_dscropper_data_loop (gpointer data)
{
  GstDsCropper *dscropper = GST_DSCROPPER (data);
  // DsCropperOutput *output;
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
      "gst-dscropper_output-loop_uid=" + std::to_string (dscropper->unique_id);

  /* Run till signalled to stop. */
  while (!dscropper->stop) {
    if (g_queue_is_empty (dscropper->data_queue)) {
      g_usleep(10000);
      continue;
    }
    
    g_mutex_lock (&dscropper->data_lock);
    
    host_ptr = g_queue_pop_head(dscropper->data_queue);
    ClippedSurfaceInfo *info = (ClippedSurfaceInfo *) g_queue_pop_head (dscropper->data_queue);
    g_mutex_unlock (&dscropper->data_lock);
    stbi_write_png(info->filename, info->width, info->height, info->channel, static_cast<unsigned char*>(host_ptr), info->pitch);
    // saveImageToRaw(image_name, host_ptr, tmp.dataSize, tmp.width, tmp.height, tmp.pitch);

    if (host_ptr){
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
dscropper_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_dscropper_debug, "dscropper", 0,
      "dscropper plugin");

  return gst_element_register (plugin, "dscropper", GST_RANK_PRIMARY,
      GST_TYPE_DSCROPPER);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_dscropper,
    DESCRIPTION, dscropper_plugin_init, "6.3", LICENSE, BINARY_PACKAGE,
    URL)

