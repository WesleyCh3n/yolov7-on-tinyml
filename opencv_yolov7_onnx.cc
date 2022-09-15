#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
#include <iostream>

template <typename T> void printv(std::vector<T> v) {
  std::cout << "[ ";
  for (T &ele : v) {
    std::cout << ele << " ";
  }
  std::cout << "]\n";
}

int main(int argc, char *argv[]) {
  Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_FATAL, "yolov7");
  Ort::SessionOptions opt = Ort::SessionOptions();
  opt.SetInterOpNumThreads(4);
  opt.SetIntraOpNumThreads(4);
  opt.SetExecutionMode(ORT_SEQUENTIAL);
  opt.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
  Ort::Session session = Ort::Session(env, "./best-320.onnx", opt);

  size_t numInputCount = session.GetInputCount();
  size_t numOutputCount = session.GetOutputCount();
  std::cout << "numInputCount: " << numInputCount << '\n';
  std::cout << "numOutputCount: " << numOutputCount << '\n';
  assert(numInputCount == 1);
  assert(numOutputCount == 1);

  int i = 0; // only 1 input
  Ort::AllocatorWithDefaultOptions allocator;
  const char *inputName = session.GetInputName(i, allocator);
  Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(i);
  std::vector<int64_t> inputShape =
      inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
  i = 0; // only 1 output tensor
  const char *outputName = session.GetOutputName(i, allocator);
  Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(i);
  std::vector<int64_t> outputShape =
      outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

  std::cout << "inputName: " << inputName << '\n';
  std::cout << "outputName: " << outputName << '\n';
  printv(inputShape);
  printv(outputShape);

  cv::Mat img = cv::imread("./test.jpg");
  cv::Mat inputMat;
  img.convertTo(inputMat, CV_32FC3, 1. / 255);         // to float
  cv::cvtColor(inputMat, inputMat, cv::COLOR_BGR2RGB); // BGR -> RGB
  cv::resize(inputMat, inputMat,
             cv::Size(inputShape[2], inputShape[3])); // resize to 320 x 320
  //// from hwc -> chw
  inputMat = inputMat.reshape(1, inputMat.rows * inputMat.cols);
  cv::transpose(inputMat, inputMat);
  //// or
  // cv::dnn::blobFromImage(inputMat, inputMat);
  std::vector<float> inputTensorValue;
  inputTensorValue.assign(inputMat.begin<float>(), inputMat.end<float>());

  std::cout << "inputMat shape: " << inputMat.size() << '\n';
  std::cout << "inputTensorValue size: " << inputTensorValue.size() << '\n';

  std::vector<const char *> inputNames{inputName};
  std::vector<const char *> outputNames{outputName};
  std::vector<Ort::Value> inputTensors;
  std::vector<Ort::Value> outputTensors;

  auto allocatorInfo =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  inputTensors.emplace_back(Ort::Value::CreateTensor<float>(
      allocatorInfo, inputTensorValue.data(), inputTensorValue.size(),
      inputShape.data(), inputShape.size()));

  double total = 0;
  for (int i = 0; i < 60; i++) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;
    start = std::chrono::system_clock::now();
    outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                                inputTensors.data(), 1, outputNames.data(), 1);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    printf("s: %.10f\r", elapsed_seconds.count());
    total += elapsed_seconds.count();
  }
  std::cout << std::endl;
  std::cout << "average: " << total / 60 << '\n';
  std::cout << "fps: " << 1 / (total / 60) << '\n';

  std::cout << "outputTensor size: " << outputTensors.size() << '\n';
  float *pData = outputTensors[0].GetTensorMutableData<float>();
  auto outputCount = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  int64_t nelem = 1;
  for (auto &i : outputCount) {
    nelem *= i;
  }
  std::vector<float> data(pData, pData + nelem);
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
    int x1 = (int)(data[id - 5] / 320 * img.cols);
    int y1 = (int)(data[id - 4] / 320 * img.rows);
    int x2 = (int)(data[id - 3] / 320 * img.cols);
    int y2 = (int)(data[id - 2] / 320 * img.rows);
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                  cv::Scalar(0, 255, 255));
  }

  cv::imwrite("./result.jpg", img);
  std::cout << "Finish\n";

  return 0;
}
