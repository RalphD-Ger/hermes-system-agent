# hermes-system-agent
Autonomous Hermes agent on NVIDIA Jetson Orin Nano for file management, system automation, and AI-powered deep coding via Anthropic API. Supports Image-Gen via XIA-API.

## Deep Coding — ML Pipeline Example

The user sends a request directly in Discord. Agent preprocesses 
it and forwards it to Claude Opus 4.8 via the Anthropic API.

![Discord Request](examples/deep-coding/screenshots/00_discord_request.png)

Agent returns the full response split across multiple messages (because of character limit per message).
Here is the PyTorch training script:

![train.py Response](examples/deep-coding/screenshots/01_response_train_1.png)
![train.py Response](examples/deep-coding/screenshots/02_response_train_2.png)

ONNX export and Python inference:

![export_onnx.py](examples/deep-coding/screenshots/03_response_export_onnx.png)
![infer_python.py](examples/deep-coding/screenshots/04_response_infer_python.png)

C++ ONNX Runtime inference with CMakeLists:

![CMakeLists](examples/deep-coding/screenshots/05_response_cpp_cmake.png)
![infer.cpp](examples/deep-coding/screenshots/06_response_cpp_infer_1.png)
![infer.cpp](examples/deep-coding/screenshots/07_response_cpp_infer_2.png)

Cost report and Anthropic Console verification. You see that Cost report & Anthropic Console show exact same Token Count, confirming correct Token & Cost Calculation:

![Cost Report](examples/deep-coding/screenshots/08_response_cpp_infer_3_costrepo.png)
![Anthropic Logs](examples/deep-coding/screenshots/09_anthropic_console_logs.png)