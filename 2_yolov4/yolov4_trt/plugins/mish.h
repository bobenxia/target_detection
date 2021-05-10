//
// Created by xia on 2021/4/28.
//

#ifndef PLUGINS_MISH_H
#define PLUGINS_MISH_H

#include <string>
#include <vector>
#include "NvInfer.h"

namespace nvinfer1
{
    class MishPlugin: public IPluginV2IOExt
    {
    public:
        explicit MishPlugin();
        MishPlugin(const void* data, size_t length);

        ~MishPlugin();

        int getNbOutputs() const override
        {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

        int initialize() override;

        virtual void terminate() override {};

        virtual
    };
}




















#endif //PLUGINS_MISH_H
