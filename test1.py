    # Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import deque

import argparse
import os
import sys
import time
import cv2

import numpy as np
import tensorflow as tf

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  file_name = "tf_files/videos"
  model_file = "tf_files/retrained_graph.pb"
  label_file = "tf_files/retrained_labels.txt"
  output = "output"
  input_height = 299
  input_width = 299
  input_mean = 128
  input_std = 128
  input_layer = "Mul"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--output", help="name of dir to write output to")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.output:
    output = args.output
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)

  for video_name in os.listdir(file_name):

      #custom variables for smoothing
      smooth_lst = deque(maxlen=10)
      smooth_var = 0

      #custom variables for counting chews
      chew_count = 0
      toggle = True

      #timing dictionary
      times = {}

      video_capture = cv2.VideoCapture(file_name+"/"+video_name)
      while (video_capture.isOpened()):

          ret,frame = video_capture.read()
          frameId = video_capture.get(1) #current frame number
          if ((frameId % 2) ==0 and (ret == True)):

              cv2.imwrite(filename="screens/"+"alpha.jpg", img=frame);
              image_data = "screens/alpha.jpg"
              t = read_tensor_from_image_file(image_data,
                                              input_height=input_height,
                                              input_width=input_width,
                                              input_mean=input_mean,
                                              input_std=input_std)

              input_name = "import/" + input_layer
              output_name = "import/" + output_layer
              input_operation = graph.get_operation_by_name(input_name);
              output_operation = graph.get_operation_by_name(output_name);

              with tf.Session(graph=graph) as sess:
                #start = time.time()
                results = sess.run(output_operation.outputs[0],
                                  {input_operation.outputs[0]: t})
                #end=time.time()
              results = np.squeeze(results)

              top_k = results.argsort()[-5:][::-1]
              labels = load_labels(label_file)

              print (labels[0], results[0])


              if (labels[0]=="chew" and results[0]>0.9):
                smooth_lst.append("1")
              else:
                smooth_lst.append("0")

              numlist = [int(x) for x in smooth_lst]
              print (numlist)

              new_frame = cv2.resize(frame, (960,540))

              if (sum(numlist)>=4 and toggle == True):

                  toggle = False
                  print ("HERE WE GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO!!")
                  print (frameId)
                  chew_count += 1

                  #populate dict with times of chews, time.time works on Unix and Windows but time.clock doesnt
                  times[chew_count] = frameId
                  print (times)
                  #print (numlist)

              if (sum(numlist)==0):
                  toggle = True


              if (toggle == False):
                   #superimpose "Chew" text on output frame
                   font = cv2.FONT_HERSHEY_SIMPLEX
                   cv2.putText(new_frame,'Chew',(10,60), font, 2,(124,252,0),2,cv2.LINE_AA)

              img_path = "D:/Jpegs"
              cv2.imwrite(img_path+"/"+str(frameId)+video_name[:-4]+".jpg", new_frame)
              #cv2.imshow("image", new_frame)
              #cv2.waitKey(30)

          if (ret == False):
              break

      if (chew_count>1):
          #access the last chew frame and subtract when the first chew frame occurred
          time = (times[chew_count]-times[1])/25
          print ("Time for 1st chew: " + str(times[1]))
          print ("This is the chewcount" + str(chew_count))
          print ("Time for last chew: " + str(times[chew_count]))

          chew_rate = time/(chew_count-1)

          #save chew_rates to files
          filepath = output + "/" + video_name[:-4]

          f = open(filepath+".txt", "w+")
          f.write(str(chew_rate))
          f.close()
          f= open(output+"/all.txt", "a+")
          f.write(str(chew_rate)+"\n")
          f.close()

          print (chew_rate)

      else:
          print ("Minimum of two chews needs to occur to calculate rate!")

          #print (smooth_lst)
      print ("Chews: " + str(chew_count))
      print (times)

      #empty out dictionary
      times.clear()

      video_capture.release()
      cv2.destroyAllWindows()
