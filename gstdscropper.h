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

#ifndef __GST_DSCROPPER_H__
#define __GST_DSCROPPER_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

/* Open CV headers */
#ifdef WITH_OPENCV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "nvtx3/nvToolsExt.h"

#include "nppi.h"
#include "nppi_geometry_transforms.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <unordered_map>
#include <list>
#include <string>
#include <errno.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
// #include "fdfs_client.h"
// #include "fdfs_global.h"
#include "cfdfs_client.h"



/* Package and library details required for plugin_init */
#define PACKAGE "dscropper"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "Gstreamer plugin for clipping objects with Deepstream nvinfer outputs"
#define BINARY_PACKAGE "Gstreamer plugin"
#define URL "https://github.com/zhouyuchong/gst-dscropper"


G_BEGIN_DECLS
/* Standard boilerplate stuff */
typedef struct _GstDsCropper GstDsCropper;
typedef struct _GstDsCropperClass GstDsCropperClass;

/* Standard boilerplate stuff */
#define GST_TYPE_DSCROPPER (gst_dscropper_get_type())
#define GST_DSCROPPER(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DSCROPPER,GstDsCropper))
#define GST_DSCROPPER_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DSCROPPER,GstDsCropperClass))
#define GST_DSCROPPER_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DSCROPPER, GstDsCropperClass))
#define GST_IS_DSCROPPER(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DSCROPPER))
#define GST_IS_DSCROPPER_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DSCROPPER))
#define GST_DSCROPPER_CAST(obj)  ((GstDsCropper *)(obj))

/** Maximum batch size to be supported by dscropper. */
#define NVDSCROPPER_MAX_BATCH_SIZE G_MAXUINT

typedef struct
{
  guint width;
  guint height;
  guint left;
  guint top;
  guint counter;
} CropperObjectInfo;


struct _GstDsCropper
{
  GstBaseTransform base_trans;

  /** Processing Queue and related synchronization structures. */

  /** Gmutex lock for against shared access in threads**/
  GMutex process_lock;

  GMutex data_lock;

  GMutex fdfs_lock;

  /** Queue to send data to output thread for processing**/
  GQueue *process_queue;

  GQueue *data_queue;

  GQueue *fdfs_queue;

  std::unordered_map<guint64, CropperObjectInfo> *object_infos;
  std::list<guint64> *insertion_order;
  
  /** Gcondition for process queue**/
  GCond process_cond;

  GCond data_cond;

  /**Queue to receive processed data from output thread **/
  GQueue *buf_queue;

  /** Gcondition for buf queue **/
  GCond buf_cond;

  /** Output thread. */
  GThread *process_thread;

  GThread *data_thread;

  /** Boolean to signal output thread to stop. */
  gboolean stop;

  /** Unique ID of the element. Used to identify metadata
   *  generated by this element. */
  guint unique_id;

  /** Frame number of the current input buffer */
  guint64 frame_num;

  /** CUDA Stream used for allocating the CUDA task */
  cudaStream_t cuda_stream;

  /** Temporary NvBufSurface for batched transformations. */
  NvBufSurface batch_insurf;

  /** the intermediate scratch buffer for conversions RGBA */
  NvBufSurface *inter_buf;

  /** Input video info (resolution, color format, framerate, etc) */
  GstVideoInfo video_info;

  /** GPU ID on which we expect to execute the task */
  guint gpu_id;

  /** Current batch number of the input batch. */
  gulong current_batch_num;

  /** GstFlowReturn returned by the latest buffer pad push. */
  GstFlowReturn last_flow_ret;

  /** NVTX Domain. */
  nvtxDomainHandle_t nvtx_domain;

  gchar *output_path;
  gchar *fdfs_cfg_path;
  // gchar *name_format;
  gint interval;
  gint operate_on_gie_id;
  std::vector<gboolean> *operate_on_class_ids;
  gfloat scale_ratio;
  gint crop_mode;

  gint save_mode;

  CFDFSClient fdfs_client;
  
};

typedef struct
{
  /** Ratio by which the frame / object crop was scaled in the horizontal
   * direction. Required when scaling co-ordinates/sizes in metadata
   * back to input resolution. */
  gdouble scale_ratio_x = 0.0;
  /** Ratio by which the frame / object crop was scaled in the vertical
   * direction. Required when scaling co-ordinates/sizes in metadata
   * back to input resolution. */
  gdouble scale_ratio_y = 0.0;
  /** NvDsObjectParams belonging to the object to be classified. */
  NvDsObjectMeta *obj_meta = nullptr;
  NvDsFrameMeta *frame_meta = nullptr;
  /** Index of the frame in the batched input GstBuffer. Not required for
   * classifiers. */
  guint batch_index = 0;
  /** Frame number of the frame from the source. */
  gulong frame_num = 0;
  /** The buffer structure the object / frame was converted from. */
  NvBufSurfaceParams *input_surf_params = nullptr;
} GstDsCropperFrame;

/**
 * Holds information about the batch of frames to be inferred.
 */
typedef struct
{
  /** Vector of frames in the batch. */
  std::vector < GstDsCropperFrame > frames;
  /** Pointer to the input GstBuffer. */
  GstBuffer *inbuf = nullptr;
  /** Batch number of the input batch. */
  gulong inbuf_batch_num = 0;
  /** Boolean indicating that the output thread should only push the buffer to
   * downstream element. If set to true, a corresponding batch has not been
   * queued at the input of NvDsExampleContext and hence dequeuing of output is
   * not required. */
  gboolean push_buffer = FALSE;
  /** Boolean marking this batch as an event marker. This is only used for
   * synchronization. The output loop does not process on the batch.
   */
  gboolean event_marker = FALSE;

#ifdef WITH_OPENCV
  /** OpenCV mat containing RGB data */
  cv::Mat * cvmat;
#else
  NvBufSurface *inter_buf;
#endif

  nvtxRangeId_t nvtx_complete_buf_range = 0;
} GstDsCropperBatch;

typedef struct
{
  guint width;
  guint height;
  // gchar *filename;
  guint track_id;
  guint frame_num;
  unsigned int source_id;
  // image type 0: object, 1: frame
  guint image_type;
} ClippedSurfaceInfo;

/** Boiler plate stuff */
struct _GstDsCropperClass
{
  GstBaseTransformClass parent_class;
};

GType gst_dscropper_get_type (void);

G_END_DECLS
#endif /* __GST_DSCROPPER_H__ */
