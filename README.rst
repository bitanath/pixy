🧚‍♂️ Pixy - Literally the Smallest Functioning LLM Ever
==========================================================

✨ Originally built for the Arm Optimization Challenge Hackathon (2026).
The project is not vibe-coded, but all Markdown and RST for textual descriptions, 
including this Readme, is generated from the source code using LLMs.


Overview
========

Pixy is a fully self-contained machine learning inference framework
written entirely in Zig, designed to run quantized language models under
extreme memory and compute constraints. Optimized using Arm Performix.

The primary demonstration is Falcon H1 0.5B running as an 8-bit
quantized model on Apple Watch hardware.

Despite the limitations of wearable hardware, Pixy achieves approximately
1.5 tokens/sec on-device while maintaining language generation
quality thanks to following  implemented from scratch in this repo.

* Tensor operations.
* Quantization and dequantization.
* Matrix multiplication kernels.
* Convolution kernels.
* ARM SIMD acceleration.
* watchOS integration.

The inference runtime has no dependency on Python, PyTorch, TensorFlow,
ONNX Runtime, or libtorch.

Documentation
=============

* `HOW-TO-LLM.md <HOW-TO-LLM.md>`_ — Agent guide covering runtime
  architecture, source organization, extending the inference engine, kernel
  development, and debugging workflows.

* `OPTIMIZATIONS.md <OPTIMIZATIONS.md>`_ — Technical notes covering Zig SIMD
  primitives, ARM NEON mappings, vectorized kernels, and quantized inference
  optimizations.


Optimizations
=============

Pixy relies heavily on Zig compile-time SIMD support for ARM NEON intrinsics.

Major techniques include:

* ``@Vector`` SIMD operations.
* ``@mulAdd`` fused multiply-add.
* ``@reduce`` horizontal reductions.
* ``@shuffle`` vector operations.
* ``@select`` branchless conditionals.
* Vectorized quantization/dequantization.
* Compile-time loop unrolling.

Detailed instruction-level analysis is available in:

::

    OPTIMIZATIONS.md

Performance Tooling
====================

Performix provides profiling utilities for continuous optimization.
However, the GUI requires a lot of context switching. We build a lightweight
CLI in zig in order to parse reports and generate symbolic information to iterate faster
for performance optiimizations.

Supported workflows:

* Code hotspot analysis.
* System utilization analysis.

The project includes benchmark Performix reports under:

::

    performix/reports/


Use the Performix CLI to analyze generated profiling reports and iterate
on performance optimizations.

::

    pxy performix/reports/code_hotspots_3ac75e10feeb.zip

or for Linux:

::

    pxy-linux performix/reports/code_hotspots_3ac75e10feeb.zip

Supported reports include code hotspots and system utilization analysis.

Prerequisites
=============

Required:

* Zig compiler.
* Git.

For watchOS builds:

* macOS.
* Xcode.
* watchOS SDK.

For training:

* Python 3.11+
* PyTorch.
* Jupyter.


Installing Zig
==============

macOS
-----

Using Homebrew:

::

    brew install zig


Verify:

::

    zig version


Linux
-----

Download Zig from:

::

    https://ziglang.org/download/


Example:

::

    tar xf zig-linux-x86_64.tar.xz

Add Zig to PATH:

::

    export PATH=$PATH:/path/to/zig


Verify:

::

    zig version


Building with Zig
=================

Pixy uses the Zig build system.

Individual components can also be built directly:

::

    zig build


Release builds:

::

    zig build -Doptimize=ReleaseFast


I have provided shell scripts ``deploy.sh`` for every target feature to make deployment simpler, but
I may have missed out for linux specific things. Usually, fallback to ``zig build`` only if the ``deploy.sh`` does not work.


Features
========

On-Device LLM Inference
-----------------------

Pixy's primary feature is running Falcon H1 0.5B as an 8-bit quantized
language model directly on Apple Watch.

Supported inference targets:

* watchOS devices.
* watchOS Simulator.
* macOS.
* Linux.

The same Zig runtime produces libraries and binaries across these
platforms.

Server-Side PII Masking Demo
----------------------------

Pixy also includes standalone inference binaries demonstrating
server-side log processing using the same model.

The core build produces:

::

    llm-linux-piimask
    llm-macos-piimask


These binaries expose Falcon H1 inference for lightweight server-side
experiments such as:

* Log sanitization.
* PII masking.
* Privacy-aware processing pipelines.

The binaries are statically linked and require no ML
runtime installation.

Visual Touch Prompting
----------------------

Pixy includes a lightweight convolutional neural network that converts
drawn Apple Watch gestures into prompts.

Example workflow:

::

    Apple Watch Drawing
            |
            v
    Touch Recognition CNN
            |
            v
    Generated Prompt
            |
            v
    Falcon H1 Inference


The CNN is trained separately using the Python training pipeline and
deployed using the Zig inference runtime.


Architecture
============

Pixy consists of four major components.

Core LLM Runtime
----------------

Location:

::

    framework/


A dependency-free Zig implementation of quantized transformer inference.

Responsibilities:

* Model execution.
* Tensor operations.
* Quantized arithmetic.
* SIMD accelerated kernels.
* Platform builds.


Visual Runtime
--------------

Location:

::

    visual/


A Zig CNN inference runtime for touch gesture recognition.

Responsibilities:

* CNN execution.
* Model loading.
* Weight conversion.
* watchOS integration.


Training Pipeline
-----------------

Location:

::

    training/


Python-based experimentation and training environment.

Contains:

* Dataset.
* PyTorch notebooks.
* CNN training scripts.
* Model checkpoints.


watchOS Application
-------------------

Locations:

::

    pixy.xcodeproj
    pixy Watch App/


Native SwiftUI application integrating:

* LLM runtime.
* Visual recognition runtime.
* Touch interface.
* Chat interface.


Repository Layout
=================

::

    .
    ├── framework/          # Zig LLM inference runtime
    │
    ├── visual/             # Zig CNN inference runtime
    │
    ├── training/           # Python/PyTorch training pipeline
    │
    ├── performix/          # Profiling and optimization tooling
    │
    ├── pixy.xcodeproj      # watchOS Xcode project
    │
    ├── OPTIMIZATIONS.md    # ARM NEON optimization details
    │
    ├── HOW-TO-LLM.md       # Developer and agent documentation
    │
    └── README.rst



Building Components
===================

Pixy is divided into multiple independently buildable components.

The deployment scripts handle platform configuration and output
generation.

Core LLM Runtime
----------------

The core runtime is located under:

::

    framework/


Build:

::

    cd framework/scripts
    ./deploy.sh


The build generates libraries and binaries under:

::

    framework/outputs/


Library outputs:

::

    framework/outputs/lib


Generated libraries target:

* watchOS.
* watchOS Simulator.
* macOS.
* Linux.


Binary outputs:

::

    framework/outputs/bin


Generated binaries include:

* Linux executables.
* macOS executables.
* Server-side demonstration tools.


Server PII Masking Demo
~~~~~~~~~~~~~~~~~~~~~~~

The core build includes standalone inference binaries:

::

    llm-linux-piimask
    llm-macos-piimask


These binaries demonstrate running the quantized Falcon H1 model for
server-side log processing.

Example use cases:

* Masking sensitive information from application logs.
* Testing local inference pipelines.
* Privacy-preserving preprocessing.


Visual Touch Recognition CNN
----------------------------

The visual runtime is located under:

::

    visual/


Build:

::

    cd visual/scripts
    ./deploy.sh


The build produces:

::

    visual/outputs/


Including:

::

    visual/outputs/lib


Targets:

* watchOS framework.
* watchOS Simulator framework.


The CNN converts Apple Watch drawings into prompts for the LLM.

Example pipeline:

::

    User Drawing
        |
        v
    Visual CNN
        |
        v
    Text Prompt
        |
        v
    Falcon H1


The visual runtime itself is implemented in Zig and does not require a
Python runtime during inference.


Performix CLI
-------------

Performix provides tooling for continuous optimization and analysis.

Build:

::

    cd performix
    ./deploy.sh


This produces the ``pxy`` command line tool.

Example usage:

::

    pxy performix/reports/code_hotspots_3ac75e10feeb.zip


or:

::

    pxy-linux performix/reports/code_hotspots_3ac75e10feeb.zip


Included reports:

::

    performix/reports/

    code_hotspots_3ac75e10feeb.zip
    code_hotspots_835a0fc8224b.zip
    code_hotspots_a2419b38d3b0.zip
    system_utilization_c7e6f3f5707c.zip


Other Performix integrations are currently unsupported because they
require additional PMU counters unavailable in the development
environment.

Example error:

::

    tool_integrations.neoprof.INSUFFICIENT_PMU_COUNTERS


watchOS Application
===================

The Apple Watch application is located in:

::

    pixy.xcodeproj

and:

::

    pixy Watch App/


Open the project:

::

    open pixy.xcodeproj


Build and deploy using Xcode.

The application integrates:

* ``libllm.a`` - Zig LLM inference runtime.
* ``libvisual.a`` - Zig CNN inference runtime.


Simulator Builds
----------------

For watchOS Simulator development:

* Build frameworks without the ``--release`` flag.
* Use simulator-compatible framework targets.


Xcode currently does not support release watchOS framework targets for
the simulator in the same way as physical Apple Watch deployments.


Training Pipeline
=================

Training utilities are located under:

::

    training/


The inference runtime does not require Python.

Python is only used for:

* CNN training.
* Dataset experimentation.
* Validation.
* Weight conversion.


Contents:

::

    training/

    dataset/
        Gesture training images.

    train_convnet.ipynb
        CNN training notebook.

    train_colab.ipynb
        Google Colab training notebook.

    weights.pth
        Generated model checkpoint.

    weights_best.pth
        Best validation checkpoint.


The dataset contains gesture classes used by the touch recognition
pipeline:

::

    angry
    confused
    cross
    flabbergasted
    happy
    heart
    question
    sad
    tick


After training, model weights can be converted into the format consumed
by the Zig inference runtime.

The visual conversion utility is:

::

    visual/convert_weights.py






License
=======

This project is licensed under the Apache License 2.0.

When used with Falcon H1 model weights, the model weights are licensed
separately under the Falcon-LLM License, which is derived from Apache 2.0.

Apache 2.0:

::

    https://www.apache.org/licenses/LICENSE-2.0


Falcon-LLM License:

::

    https://falconllm.tii.ae/falcon-terms-and-conditions.html