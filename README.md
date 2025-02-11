<!--
 * @Author: zhouyuchong
 * @Date: 2024-10-24 16:58:01
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2025-02-11 14:55:41
-->
# gst-dscropper

crop & save images base on model outputs. 

## feature
+ Non-block data thread

## property
### interval(int)
+ -1: only cropping on the first time this object is detected.
+ n: crop every n frames.

### name-format
keywords: frame, trackid, classid, conf

e.g.:
```
set_property("name-format", "trackid;frame;conf")
```
output file name: `tida_frmb_confc.png`


for more details, run `gst-inspect-1.0 dscropper`

## Usage
+ install fastdfs, [tutorial](https://blog.csdn.net/qq_41453285/article/details/107158911)
+ export CUDA_VER=?
+ make install

## Reference
+ gst-dsexample
+ [stb](https://github.com/nothings/stb)
