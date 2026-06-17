#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

constexpr int64_t kH = 28, kW = 28, kClasses = 10;

std::vector<float> load_image(const char* path) {
    std::vector<float> img(kH * kW, 0.0f);
    if (!path) return img;

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Warning: cannot open " << path << ", using zeros.\n";
        return img;
    }
    f.read(reinterpret_cast<char*>(img.data()),
           static_cast<std::streamsize>(img.size() * sizeof(float)));
    if (f.gcount() != static_cast<std::streamsize>(img.size() * sizeof(float))) {
        std::cerr << "Warning: short read, using zeros.\n";
        std::fill(img.begin(), img.end(), 0.0f);
    }
    return img;
}

void softmax(std::vector<float>& v) {
    const float m = *std::max_element(v.begin(), v.end());
    float sum = 0.0f;
    for (float& x : v) { x = std::exp(x - m); sum += x; }
    for (float& x : v) x /= sum;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> [image_28x28.bin]\n";
        return 1;
    }
    const char* model_path = argv[1];
    const char* image_path = (argc >= 3) ? argv[2] : nullptr;

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mnist");
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(std::max(1u, std::thread::hardware_concurrency()));
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        std::wstring wpath(model_path, model_path + std::strlen(model_path));
        Ort::Session session(env, wpath.c_str(), opts);
#else
        Ort::Session session(env, model_path, opts);
#endif

        Ort::AllocatorWithDefaultOptions alloc;
        Ort::AllocatedStringPtr in_name = session.GetInputNameAllocated(0, alloc);
        Ort::AllocatedStringPtr out_name = session.GetOutputNameAllocated(0, alloc);
        const char* input_names[]  = {in_name.get()};
        const char* output_names[] = {out_name.get()};

        std::vector<float> img = load_image(image_path);
        std::array<int64_t, 4> in_shape{1, 1, kH, kW};

        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem, img.data(), img.size(), in_shape.data(), in_shape.size());

        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        const float* logits = outputs.front().GetTensorData<float>();
        std::vector<float> probs(logits, logits + kClasses);
        softmax(probs);

        int best = static_cast<int>(
            std::max_element(probs.begin(), probs.end()) - probs.begin());

        std::cout << "Predicted class: " << best
                  << "  (p=" << probs[best] << ")\n";
        std::cout << "Probabilities:";
        for (int i = 0; i < kClasses; ++i)
            std::cout << " " << i << ":" << probs[i];
        std::cout << "\n";

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 2;
    }
    return 0;
}