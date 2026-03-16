/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-08-21 14:06:38
 * @LastEditTime: 2020-06-10 11:51:09
 * @LastEditors: zerollzeng
 */
#include <dlfcn.h>
#include "IluvatarIxRT.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "util.h"

using namespace std;
using namespace nvinfer1;

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

Trt::Trt(std::string plugin_lib_path)
{
    m_minDim = {};
    m_optDim = {};
    m_maxDim = {};

    if (not plugin_lib_path.empty())
    {
        plugin_handle = dlopen(plugin_lib_path.c_str(), RTLD_LAZY);
        logger->info("[{} {}]: load plugin: {} !!!", __FUNCTION__, __LINE__, plugin_lib_path);
        assert(plugin_handle != nullptr && "Invaid plugin lib path ");
    }

    initLibNvInferPlugins(&mLogger, "");

    mBuilder = nvinfer1::createInferBuilder(mLogger);
    mConfig  = mBuilder->createBuilderConfig();
    mProfile = mBuilder->createOptimizationProfile();
    assert(mProfile != nullptr && "create trt builder optimazation profile failed");
}

Trt::~Trt()
{
    if (mContext != nullptr)
    {
        delete mContext;
        mContext = nullptr;
    }
    if (mEngine != nullptr)
    {
        delete mEngine;
        mEngine = nullptr;
    }
    if (mConfig != nullptr)
    {
        delete mConfig;
        mConfig = nullptr;
    }
    if (mProfile != nullptr)
    {
        mProfile = nullptr;
    }
    if (mBuilder != nullptr)
    {
        delete mBuilder;
        mBuilder = nullptr;
    }
    if (mRuntime != nullptr)
    {
        delete mRuntime;
        mRuntime = nullptr;
    }
    if (mNetwork != nullptr)
    {
        delete mNetwork;
        mNetwork = nullptr;
    }
    if (plugin_handle != nullptr)
    {
        dlclose(plugin_handle);
    }

    for (size_t i = 0; i < mBindingPtr.size(); i++)
    {
        safeCudaFree(mBindingPtr[i]);
    }
}

void Trt::CreateEngine(const std::string&              onnxModel,
                       const std::string&              engineFile,
                       const std::vector<std::string>& customOutput,
                       int                             mode)
{
    mRunMode = mode;
    if (!DeserializeEngine(engineFile))
    {
        if (!BuildEngineWithOnnx(onnxModel, engineFile, customOutput))
        {
            logger->error("[{} {}]: could not deserialize or build engine !!! ", __FUNCTION__, __LINE__);
            return;
        }
    }
    logger->info("[{} {}]: create execute context and malloc device memory... ", __FUNCTION__, __LINE__);
    InitEngine();
}

void Trt::Forward(void** buffer, int batchSize, cudaStream_t stream)
{
    if (mIsDynamicShape)
        this->SetBindingDimensions(batchSize);

    if (stream)
    {
        mContext->enqueueV2(buffer, stream, nullptr);
    }
    else
    {
        mContext->executeV2(buffer);
    }
}

void Trt::SetBindingDimensions(int batchSize)
{
    for (int i = 0; i < nbBindings; i++)
    {
        if (mEngine->bindingIsInput(i))
        {
            nvinfer1::Dims  dims_i(mEngine->getBindingDimensions(i));
            nvinfer1::Dims4 inputDims{batchSize, dims_i.d[1], dims_i.d[2], dims_i.d[3]};
            mContext->setBindingDimensions(i, inputDims);
        }
    }
}

void Trt::SetDevice(int device)
{
    checkCudaErrors(cudaSetDevice(device));
}

int Trt::GetDevice() const
{
    int device = -1;
    checkCudaErrors(cudaGetDevice(&device));
    if (device != -1)
    {
        return device;
    }
    else
    {
        logger->error("[{} {}]: Get Device Error", __FUNCTION__, __LINE__);
        return -1;
    }
}

void* Trt::GetBindingPtr(int bindIndex) const
{
    return mBindingPtr[bindIndex];
}

size_t Trt::GetBindingSize(int bindIndex) const
{
    return mBindingSize[bindIndex];
}

nvinfer1::Dims Trt::GetBindingDims(int bindIndex) const
{
    return mBindingDims[bindIndex];
}

std::vector<int> Trt::GetBindingDimsVec(int bindIndex) const
{
    std::vector<int> dims;
    for (size_t j = 0; j < mBindingDims[bindIndex].nbDims; j++)
    {
        dims.push_back(mBindingDims[bindIndex].d[j]);
    }
    return dims;
}

nvinfer1::DataType Trt::GetBindingDataType(int bindIndex) const
{
    return mBindingDataType[bindIndex];
}

std::string Trt::GetBindingName(int bindIndex) const
{
    return mBindingName[bindIndex];
}

int Trt::GetNbInputBindings() const
{
    return mNbInputBindings;
}

int Trt::GetNbOutputBindings() const
{
    return mNbOutputBindings;
}

bool Trt::DeserializeEngine(const std::string& engineFile)
{
    std::ifstream input(engineFile, std::ios::ate | std::ios::binary);
    std::vector<char> buffer;
    if (input.is_open())
    {
        logger->info("[{} {}]: deserialize engine from {}", __FUNCTION__, __LINE__, engineFile);
        std::streamsize size = input.tellg();
        input.seekg(0, std::ios::beg);
        buffer.resize(size);
        std::vector<char> *raw_plan = &buffer;
        input.read(raw_plan->data(), size);
        mRuntime = nvinfer1::createInferRuntime(mLogger);
        mEngine  = mRuntime->deserializeCudaEngine(raw_plan->data(), raw_plan->size());
        assert(mEngine != nullptr);
        if (mIsDynamicShape)
        {
            assert(mProfile->isValid() && "Invalid dynamic shape profile");
            mConfig->addOptimizationProfile(mProfile);
        }
        return true;
    }
    return false;
}

void Trt::AddDynamicShapeProfile(const std::string&      inputName,
                                 const std::vector<int>& minDimVec,
                                 const std::vector<int>& optDimVec,
                                 const std::vector<int>& maxDimVec)
{
    nvinfer1::Dims minDim, optDim, maxDim;
    int            nbDims = optDimVec.size();
    minDim.nbDims         = nbDims;
    optDim.nbDims         = nbDims;
    maxDim.nbDims         = nbDims;
    for (int i = 0; i < nbDims; i++)
    {
        minDim.d[i] = minDimVec[i];
        optDim.d[i] = optDimVec[i];
        maxDim.d[i] = maxDimVec[i];
    }
    mProfile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMIN, minDim);
    mProfile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kOPT, optDim);
    mProfile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMAX, maxDim);
    mIsDynamicShape = true;
    mBatchSize      = maxDimVec[0];
}

void Trt::BuildEngine(const std::string& fileName)
{
    if (mRunMode == 1)
    {
        mConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
        logger->info("[{} {}]: BuilderFlag: kFP16", __FUNCTION__, __LINE__);
    }
    else
    {
        mConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
        logger->info("[{} {}]: BuilderFlag: kINT8", __FUNCTION__, __LINE__);
    }
    cout << "build engine..." << endl;
    mRuntime = nvinfer1::createInferRuntime(mLogger);
    unique_ptr<nvinfer1::IHostMemory> data{mBuilder->buildSerializedNetwork(*mNetwork, *mConfig)};
    if (not data)
    {
        logger->error("[{} {}]: Create serialized engine plan failed", __FUNCTION__, __LINE__);
        return;
    }
    else
    {
        logger->info("[{} {}]: Create serialized engine plan done", __FUNCTION__, __LINE__);
    }
    if (fileName == "")
    {
        logger->warn("[{} {}]: empty engine file name, skip save", __FUNCTION__, __LINE__);
        return;
    }
    logger->info("[{} {}]: save engine to {} ", __FUNCTION__, __LINE__, fileName);
    std::ofstream file;
    file.open(fileName, std::ios::binary | std::ios::out);
    if (!file.is_open())
    {
        logger->error("[{} {}]: read create engine file {} failed", __FUNCTION__, __LINE__, fileName);
        return;
    }
    file.write((const char*)data->data(), data->size());
    file.close();

    mEngine = mRuntime->deserializeCudaEngine(data->data(), data->size());
    assert(mEngine != nullptr);
}

bool Trt::BuildEngineWithOnnx(const std::string&              onnxModel,
                              const std::string&              engineFile,
                              const std::vector<std::string>& customOutput)
{
    logger->info("[{} {}]: build onnx engine from {}... ", __FUNCTION__, __LINE__, onnxModel);
    assert(mBuilder != nullptr);
    mFlags   = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    mNetwork = mBuilder->createNetworkV2(mFlags);
    assert(mNetwork != nullptr);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*mNetwork, mLogger);
    if (!parser->parseFromFile(onnxModel.c_str(), static_cast<int>(mLogger.getReportableSeverity())))
    {
        logger->error("[{} {}]: could not parse onnx engine", __FUNCTION__, __LINE__);
        return false;
    }
#ifdef DEBUG
    for (int i = 0; i < mNetwork->getNbLayers(); i++)
    {
        nvinfer1::ILayer* custom_output = mNetwork->getLayer(i);
        for (int j = 0; j < custom_output->getNbInputs(); j++)
        {
            nvinfer1::ITensor* input_tensor = custom_output->getInput(j);
            logger->debug("[{} {}]: input tensor -> {}", __FUNCTION__, __LINE__, input_tensor->getName());
        }
        for (int j = 0; j < custom_output->getNbOutputs(); j++)
        {
            nvinfer1::ITensor* output_tensor = custom_output->getOutput(j);
            logger->debug("[{} {}]: output tensor -> {}", __FUNCTION__, __LINE__, output_tensor->getName());
        }
        std::cout << std::endl;
    }
#endif
    auto num_input = mNetwork->getNbInputs();
    logger->info("[{} {}]: number of input: {}", __FUNCTION__, __LINE__, num_input);
    auto num_output = mNetwork->getNbOutputs();
    logger->info("[{} {}]: number of output: {}", __FUNCTION__, __LINE__, num_output);

    for (size_t input_index = 0; input_index < num_input; input_index++)
    {
        std::ostringstream oss;
        nvinfer1::Dims     inputDims = mNetwork->getInput(input_index)->getDimensions();
        oss << "Input" << input_index << " dims: ";
        for (auto i = 0; i < inputDims.nbDims; ++i)
        {
            oss << inputDims.d[i] << "x";
        }
        logger->info("[{} {}]: {}", __FUNCTION__, __LINE__, oss.str());
    }

    for (size_t output_index = 0; output_index < num_output; output_index++)
    {
        std::ostringstream oss;
        nvinfer1::Dims     outputDims = mNetwork->getOutput(output_index)->getDimensions();
        oss << "\nOutput" << output_index << " dims: ";
        for (auto i = 0; i < outputDims.nbDims; ++i)
        {
            oss << outputDims.d[i] << " ";
        }
        logger->info("[{} {}]: {}", __FUNCTION__, __LINE__, oss.str());
    }

    if (mIsDynamicShape)
    {
        assert(mProfile->isValid() && "Invalid dynamic shape profile");
        mConfig->addOptimizationProfile(mProfile);
    }
    else
    {
        nvinfer1::Dims inputDims = mNetwork->getInput(0)->getDimensions();
        mBatchSize               = inputDims.d[0];
    }

    BuildEngine(engineFile);

    delete parser;
    return true;
}

void Trt::InitEngine()
{
    logger->info("[{} {}]: init engine...", __FUNCTION__, __LINE__);
    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);

    logger->info("[{} {}]: malloc device memory...", __FUNCTION__, __LINE__);
    nbBindings = mEngine->getNbBindings();
    logger->info("[{} {}]: nbBingdings: {}", __FUNCTION__, __LINE__, nbBindings);

    // Auto-detect dynamic-shape engine: if any input binding reports -1 in
    // the batch dimension, the deserialized engine is dynamic.  This handles
    // engines that were loaded via DeserializeEngine (which does not call
    // AddDynamicShapeProfile, so mIsDynamicShape stays false by default).
    if (!mIsDynamicShape)
    {
        for (int i = 0; i < nbBindings; i++)
        {
            if (mEngine->bindingIsInput(i))
            {
                auto dims = mEngine->getBindingDimensions(i);
                if (dims.nbDims > 0 && dims.d[0] == -1)
                {
                    mIsDynamicShape = true;
                    logger->info("[{} {}]: dynamic-shape engine detected automatically",
                                 __FUNCTION__, __LINE__);
                    break;
                }
            }
        }
    }

    // For dynamic-shape engines, set the execution context to the maximum
    // batch size so that output binding dimensions become concrete before we
    // calculate allocation sizes.
    if (mIsDynamicShape)
        SetBindingDimensions(mBatchSize);

    mBindingSize.resize(nbBindings);
    mBindingName.resize(nbBindings);
    mBindingDims.resize(nbBindings);
    mBindingDataType.resize(nbBindings);
    mBindingPtr.resize(nbBindings);
    for (int i = 0; i < nbBindings; i++)
    {
        std::ostringstream oss;
        // For dynamic-shape engines use the context dims (concrete after
        // SetBindingDimensions), so output shapes like [batch, max_boxes, 4]
        // are fully resolved before we compute the allocation size.
        nvinfer1::Dims     dims  = mIsDynamicShape ?
                                   mContext->getBindingDimensions(i) :
                                   mEngine->getBindingDimensions(i);
        nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
        const char*        name  = mEngine->getBindingName(i);
        int64_t            totalSize = 0;
        if (mIsDynamicShape)
            // dims are now concrete (e.g. [32, 1000, 4]); divide by batch to
            // get the per-image size stored in mBindingSize.
            totalSize = volume(dims) * getElementSize(dtype) / dims.d[0];
        else
            totalSize = volume(dims) * getElementSize(dtype);
        mBindingSize[i]     = totalSize;
        mBindingName[i]     = name;
        mBindingDims[i]     = dims;
        mBindingDataType[i] = dtype;
        if (mEngine->bindingIsInput(i))
        {
            oss << "input ";
        }
        else
        {
            oss << "output ";
        }
        oss << i << ": name: {" << name << "}, size in byte: {" << totalSize << "}, ";
        oss << dims.nbDims << " dimemsion: ";
        for (int j = 0; j < dims.nbDims; j++)
        {
            oss << dims.d[j] << " x ";
        }
        oss << "\b\b";
        size_t maxSize = static_cast<size_t>(totalSize) * static_cast<size_t>(mBatchSize);
        mBindingPtr[i] = safeCudaMalloc(maxSize);

        if (mEngine->bindingIsInput(i))
        {
            mNbInputBindings++;
            oss << ", addr:{" << mBindingPtr[i] << "}";
        }
        else
        {
            mNbOutputBindings++;
            oss << ", addr:{" << mBindingPtr[i] << "}";
        }
        logger->info("[{} {}]: {}", __FUNCTION__, __LINE__, oss.str());
    }
    logger->info("[{} {}]: init finish\n", __FUNCTION__, __LINE__);
}
