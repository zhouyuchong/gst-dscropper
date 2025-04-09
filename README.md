<!--
 * @Author: zhouyuchong
 * @Date: 2024-10-24 16:58:01
 * @Description: 
 * @LastEditors: zhouyuchong
 * @LastEditTime: 2025-04-03 15:32:06
-->
# gst-dscropper

crop & save images base on model outputs. 

## feature
+ Non-block data thread
+ opencv free

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

## Reference
+ gst-dsexample
+ [stb](https://github.com/nothings/stb)

1. 该帧中有满足条件的物体，才触发保存策略。触发后才看是保存帧还是保存物体还是二者都保存
