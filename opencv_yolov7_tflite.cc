#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include <iostream>

int main(int argc, char *argv[]) {
  // initailize tflite model
  const char *filename = "./best.tflite";
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
  interpreter->AllocateTensors();

  // get input & output layer
  TfLiteTensor *input_tensor = interpreter->tensor(interpreter->inputs()[0]);
  TfLiteTensor *output_tensor = interpreter->tensor(interpreter->outputs()[0]);

  // get input tensor info
  std::cout << "input dimension: " << input_tensor->dims->size << '\n';
  assert(input_tensor->dims->size == 4);              // [1, 3, 640, 640]
  std::size_t channels = input_tensor->dims->data[1]; // 3
  std::size_t rows = input_tensor->dims->data[2];     // 640
  std::size_t cols = input_tensor->dims->data[3];     // 640

  // preprocess image
  cv::Mat img = cv::imread("./test.jpg");
  cv::Mat inputImg;
  img.convertTo(inputImg, CV_32FC3);                    // to float
  cv::cvtColor(inputImg, inputImg, cv::COLOR_BGR2RGB);  // BGR -> RGB
  cv::resize(inputImg, inputImg, cv::Size(rows, cols)); // resize to 640 x 640
  inputImg = inputImg / 255.;                           // normalize
  // from h, w, c -> c, h, w
  inputImg = inputImg.reshape(1, inputImg.rows * inputImg.cols);
  cv::transpose(inputImg, inputImg);

  // fill image to input tensor
  float *inputImg_ptr = inputImg.ptr<float>(0);
  memcpy(input_tensor->data.f, inputImg.ptr<float>(0),
         channels * rows * cols * sizeof(float));

  // compute model instance
  interpreter->Invoke();

  // print final result
  std::cout << "output dimension: " << output_tensor->dims->size
            << '\n'; // 2: [100, 7]
  // count result number to fill in 1d vector
  int n = 1;
  for (int i = 0; i < output_tensor->dims->size; ++i)
    n *= output_tensor->dims->data[i];
  std::vector<float> data(output_tensor->data.f, output_tensor->data.f + n);

  // lambda to filter out low confidence result
  std::size_t idx = 0;
  auto confCondition = [&idx](const float &i) {
    if (idx++ % 7 == 6) // only check the confidence column, which is every 7th
      return i > 0.6;   // set confidence > 0.6
    else
      return false;
  };
  // get the index of condition
  std::vector<std::size_t> ids;
  auto pos = std::find_if(data.begin(), data.end(), confCondition);
  while (pos != data.end()) {
    ids.emplace_back(std::distance(data.begin(), pos));
    pos = std::find_if(std::next(pos), data.end(), confCondition);
  }

  // draw bbox
  for (std::size_t &id : ids) {
    // remap box to original image size
    int x1 = (int)(data[id - 5] / cols * img.cols);
    int y1 = (int)(data[id - 4] / rows * img.rows);
    int x2 = (int)(data[id - 3] / cols * img.cols);
    int y2 = (int)(data[id - 2] / rows * img.rows);
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                  cv::Scalar(0, 255, 255));
  }

  cv::imwrite("./result.jpg", img);
  std::cout << "Finish\n";
  return 0;
}
