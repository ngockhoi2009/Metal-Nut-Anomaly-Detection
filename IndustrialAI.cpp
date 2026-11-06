#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <vector>
#include <array>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace Ort;

int main() {
    try {
        Env env(ORT_LOGGING_LEVEL_WARNING, "PatchCore");
        SessionOptions sessionop;
        Session session(env, L"patchcore_backbone.onnx", sessionop);
        cout << "ONNX model load xong" << "\n";
        AllocatorWithDefaultOptions allocator;
        AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
        vector<int64_t> inputshape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        for (size_t i = 0; i < session.GetOutputCount(); i++) {
            AllocatedStringPtr name = session.GetOutputNameAllocated(i, allocator);
            vector<int64_t> shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        }
        vector<float>membank(3459072);
        ifstream file("memory_bank.bin", ios::binary);
        if (file.is_open()) cout << "Mo duoc file" << '\n';
        else {
            cout << "khong mo dc file";
            return 0;
        }
        file.read(reinterpret_cast<char*>(&membank[0]), 3459072 * sizeof(float));
        file.close();
        while (true) {
            cout << "Type link: ";
            string link; getline(cin, link);
            Mat image = imread(link);
            if (image.empty()) {
                cout << "Khong co anh" << '\n';
                continue;
            }
            Mat imagergb;
            cvtColor(image, imagergb, COLOR_BGR2RGB);
            imagergb.convertTo(imagergb, CV_32FC3, 1.0 / 255.0);
            Mat imageresize;
            resize(imagergb, imageresize, Size(256, 256), 0, 0, INTER_LINEAR);
            Scalar mean(0.485, 0.456, 0.406);
            Scalar std(0.229, 0.224, 0.225);
            subtract(imageresize, mean, imageresize);
            divide(imageresize, std, imageresize);
            vector<Mat> tachra;
            split(imageresize, tachra);
            int W = imageresize.cols;
            int H = imageresize.rows;
            vector<float> CHW(3 * W * H);
            int n = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < H; j++) {
                    for (int k = 0; k < W; k++) {
                        CHW[n] = tachra[i].at<float>(j, k);
                        n++;
                    }
                }
            }
            array<int64_t, 4> imagefinal = { 1, 3, H, W };
            MemoryInfo meminfo = MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Value tensor = Value::CreateTensor<float>(meminfo, CHW.data(), CHW.size(), imagefinal.data(), imagefinal.size());
            array<const char*, 1> inputname = { "input" };
            array<const char*, 2> outputname = { "layer2", "layer3" };
            RunOptions runop{ nullptr };
            vector<Value> Output = session.Run(runop, inputname.data(), &tensor, 1, outputname.data(), 2);
            float* layer2data = Output[0].GetTensorMutableData<float>();
            float* layer3data = Output[1].GetTensorMutableData<float>();
            vector<Mat> layer2;
            vector<Mat> layer3;
            for (int i = 0; i < 512; i++) {
                Mat channel(32, 32, CV_32F, layer2data + i * 1024);
                layer2.push_back(channel);
            }
            for (int i = 0; i < 1024; i++) {
                Mat channel(16, 16, CV_32F, layer3data + i * 256);
                layer3.push_back(channel);
            }
            vector<Mat> layer3resize;
            for (int i = 0; i < 1024; i++) {
                Mat temp;
                resize(layer3[i], temp, Size(32, 32));
                layer3resize.push_back(temp);
            }
            vector<vector<float>> all(1024);
            for (int i = 0; i < 1024; i++) {
                int m = i / 32;
                int n = i % 32;
                for (int j = 0; j < 512; j++) all[i].push_back(layer2[j].at<float>(m, n));
                for (int k = 0; k < 1024; k++) all[i].push_back(layer3resize[k].at<float>(m, n));
            }
            vector<float> neighbor(1024);
            #pragma omp parallel for
            for (int i = 0; i < 1024; i++) {
                float mindis = 9999999999;
                for (int j = 0; j < 2252; j++) {
                    float temp = 0;                  
                    for (int k = 0; k < 1536; k++) {
                        float dif = all[i][k] - membank[j * 1536 + k];
                        temp = temp + dif * dif;
                    }
                    if (temp < mindis) mindis = temp;
                }
                neighbor[i] = sqrt(mindis);
            }
            vector<float> top = neighbor;
            partial_sort(top.begin(), top.begin() + 10, top.end(), greater<float>());
            double imageScore = 0.0;
            for (int i = 0; i < 10; i++) imageScore = imageScore + top[i];
            imageScore = imageScore/10.0;
            cout << "Image Score: " << imageScore << '\n';
            Mat anomaly(32, 32, CV_32F);
            for (int i = 0; i < 1024; i++) {
                int j = i / 32;
                int k = i % 32;
                anomaly.at<float>(j, k) = neighbor[i];
            }
            Mat anomaly256;
            resize(anomaly, anomaly256, Size(256, 256), 0, 0, INTER_NEAREST);
            Mat anomaly256blur;
            GaussianBlur(anomaly256, anomaly256blur, Size(33, 33), 4, 4, BORDER_REFLECT);
            Mat anomalynormal;
            normalize(anomaly256blur, anomalynormal, 0, 255, NORM_MINMAX, CV_8U);
            Mat anomalycl;
            applyColorMap(anomalynormal, anomalycl, COLORMAP_JET);
            Mat image256;
            resize(image, image256, Size(256, 256));
            Mat anomalyoverlay;
            addWeighted(image256, 0.5, anomalycl, 0.5, 0, anomalyoverlay);
            constexpr float threshold = 76.6706;
            cout << "Threshold = " << threshold << '\n';
            cout << "RESULT: ";
            if (imageScore > threshold) cout << "ANOMALY" << '\n';
            else cout << "GOOD" << '\n';
            imshow("Overlay", anomalyoverlay);
            imshow("Original", image256);
            imshow("Fullheat", anomalycl);
            waitKey(0);
            destroyAllWindows();
        }
    }
    catch (const Ort::Exception& e) {
        cerr << "ONNX error: " << e.what() << '\n';
    }
    return 0;
}


