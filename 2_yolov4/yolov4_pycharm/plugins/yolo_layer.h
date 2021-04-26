#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include "math_constants.h"
#include "NvInfer.h"

#define MAX_ANCHORS 6

#define CHECK(status)                                           \
    do {                                                        \
        auto ret = status;                                      \
        if (ret != 0) {                                         \
            std::cerr << "Cuda failure in file '" << __FILE__   \
                      << "' line " << __LINE__                  \
                      << ": " << ret << std::endl;              \
            abort();                                            \
        }                                                       \
    } while (0)

namespace Yolo
{
    static constexpr float IGNORE_THRESH = 0.01f;

    struct alignas(float) Detection {
        float bbox[4];  // x, y, w, h
        float det_confidence;
        float class_id;
        float class_confidence;
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin: public IPluginV2IOExt
    {
        public:
            // parse 阶段，构造函数
            YoloLayerPlugin(int yolo_width, int yolo_height, int num_anchors, float* anchors, int num_classes, int input_width, int input_height, float scale_x_y, int new_coords);
            // deserialize 阶段，构造函数
            YoloLayerPlugin(const void* data, size_t length);

            // 析构函数
            ~YoloLayerPlugin() override = default;

            // 插件返回 tensor 数量
            int getNbOutputs() const override
            {
                return 1;
            }

            // 根据输入维度推理出模型的输出维度
            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

            // 初始化函数，在插件准备run之前执行。主要初始化一些开辟空间的参数。
            int initialize() override;

            // 释放 op 开辟的显存空间
            void terminate() override;

            // 返回这个插件 op 需要显存的实际数据大小，通过 tensorrt 的接口获取
            virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

            // 实际插件 op 的执行函数，我们自己实现的 cuda 操作就放在这个
            virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

            // 返回序列化需要写多少字节到 buffer 中
            virtual size_t getSerializationSize() const override;

            // 把需要用的数据按照顺序序列化到 buffer 里头
            virtual void serialize(void* buffer) const override;

            // 调用此方法以判断pos索引的输入/输出是否支持`inOut[pos].format`和`inOut[pos].type`指定的格式/数据类型。
            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }

            const char* getPluginType() const override;

            const char* getPluginVersion() const override;

            void destroy() override;

            // 将这个`plugin`对象克隆一份给TensorRT的builder、network或者engine。应该还有一个构造函数的？
            IPluginV2IOExt* clone() const override;

            // 为这个插件设置namespace名字，如果不设置则默认是`""`，需要注意的是同一个`namespace`下的plugin如果名字相同会冲突。
            void setPluginNamespace(const char* pluginNamespace) override;

            // 获得插件 op 的名字
            const char* getPluginNamespace() const override;

            // 返回结果类型，一般来说我们插件 op 返回结果类型与输入类型一致
            DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const override;

            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

            bool canBroadcastInputAcrossBatch(int inputIndex) const override;

            // 如果这个op使用到了一些其他东西，例如`cublas handle`，可以直接借助TensorRT内部提供的`cublas handle`:
            void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

            // 配置这个插件 op，判断输入和输出类型数量是否正确。官方还提到通过这个配置信息可以告知 TensorRT 去选择合适的算法（algorithm）去调优这个模型。
            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override TRTNOEXCEPT;

            void detachFromContext() override;

        private:
            void forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize = 1);

            int mThreadCount = 64;
            int mYoloWidth, mYoloHeight, mNumAnchors;
            float mAnchorsHost[MAX_ANCHORS * 2];
            float *mAnchors;  // allocated on GPU
            int mNumClasses;
            int mInputWidth, mInputHeight;
            float mScaleXY;
            int mNewCoords = 0;

            const char* mPluginNamespace;

        protected:
            using IPluginV2IOExt::configurePlugin;
    };

    class YoloPluginCreator : public IPluginCreator
    {
        public:
            //创建一个空的 `mPluginAtrributes` 初始化 `mFC`
            YoloPluginCreator();

            ~YoloPluginCreator() override = default;

            const char* getPluginName() const override;

            const char* getPluginVersion() const override;

            const PluginFieldCollection* getFieldNames() override;

            // 这个成员函数作用是通过 `PluginFieldCollection` 去创建 plugin，将 op 需要的权重和参数一个个取出来，然后调用中的第一个构造函数 `YoloLayerPlugin` 去创建 plugin
            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

            // 这个函数会被 `onnx-tensorrt` 的一个叫做 `TRT_PluginV2` 的转换 op 调用，这个 op 会读取 onnx 模型的 `data` 数据将其反序列化到 network 中。
            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

            void setPluginNamespace(const char* libNamespace) override
            {
                mNamespace = libNamespace;
            }

            const char* getPluginNamespace() const override
            {
                return mNamespace.c_str();
            }

        private:
            static PluginFieldCollection mFC;
            static std::vector<nvinfer1::PluginField> mPluginAttributes;
            std::string mNamespace;
    };
};

#endif
