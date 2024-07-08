#include <iostream>
#include <CL/cl.h>
#include <fstream>
#include <sstream>
#include <random>

//输入矩阵长宽
const unsigned int inputSignalWidth  = 1024;
const unsigned int inputSignalHeight = 1024;
 
//矩阵值

cl_int inputSignal[inputSignalWidth][inputSignalHeight] =
{
    {0}
};

 

 
const unsigned int maskWidth  = 8;
const unsigned int maskHeight = 8;

cl_int mask[maskWidth][maskHeight] =
{
    {0}
};

//输出长宽
const unsigned int outputSignalWidth  = inputSignalWidth - maskWidth + 1;
const unsigned int outputSignalHeight = inputSignalHeight - maskHeight + 1;

//输出矩阵值
cl_float outputSignal[outputSignalWidth][outputSignalHeight];
 /*
  1 选择平台创建OpenCL上下文
  2 选择设备创建命令队列
  3 加载内核文件(hello_world.cl)并构建到程序对象中
  4 为内核函数创建hello_kernel()创建一个内核对象
  5 为内核参数创建内存对象(result,a,b)
  6 将待执行的内核排队
  7 将内核结果读回结果缓冲区
  */
cl_context CreatContext();
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device);
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName);
bool CreateMemObjects(cl_context context, cl_mem memObjects[3], cl_int (*a)[inputSignalWidth], cl_int (*b)[maskWidth]);
void Cleanup(cl_context context, cl_command_queue commandQueue,cl_program program, cl_kernel kernel, cl_mem memObjects[3]);

 
int main(int argc, const char * argv[]) {
 
    cl_context context=0;
    cl_command_queue commandQueue=0;
    cl_program program=0;
    cl_device_id device=0;
    cl_kernel kernel=0;
    cl_mem memObjects[3]={0,0,0};
    cl_int errNum;
    
    //创建矩阵

    //获取随机数种子
    std::random_device rd;  
    //以 rd() 初始化
    std::mt19937 gen(rd()); 
    //定义分布范围[0, 1000]
    std::uniform_int_distribution<> distrib(0, 1000); 

    //打开一个文件以写入数据
    std::ofstream outfile("array.txt");

    //检查文件是否成功打开
    if (!outfile) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }

    //初始化数组
    for (int i = 0; i < inputSignalWidth; i++) {
        for (int j = 0; j < inputSignalHeight; j++) {
            //生成随机数
            inputSignal[i][j] = distrib(gen);
            outfile <<inputSignal[i][j];
            if (j != inputSignalHeight - 1) {
                outfile << ",";
            }
        }
        outfile << std::endl;
    }
    //关闭文件
    outfile.close();
    std::cout << "数组已成功保存到文件array.txt" << std::endl;
    //打印数组验证保存是否正确
    for (int i = 0; i < inputSignalWidth; i++) {
        for (int j = 0; j < inputSignalHeight; j++) {
            std::cout <<inputSignal[i][j]<<" ";
        }
        std::cout << std::endl;
    }
    std::cout << "************************"<< std::endl;


    //打开一个文件以写入数据
    std::ofstream outfile2("kernel.txt");

    //检查文件是否成功打开
    if (!outfile2) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }

    //初始化数组
    for (int i = 0; i < maskWidth; i++) {
        for (int j = 0; j < maskHeight; j++) {
            //生成随机数
            mask[i][j] = distrib(gen);
            outfile2 <<mask[i][j];
            if (j != maskHeight - 1) {
                outfile2 << ",";
            }
        }
        outfile2 << std::endl;
    }
    //关闭文件
    outfile2.close();
    std::cout << "数组已成功保存到文件kernel.txt" << std::endl;
    //打印数组验证保存是否正确
    for (int i = 0; i < maskWidth; i++) {
        for (int j = 0; j < maskHeight; j++) {
            std::cout <<mask[i][j]<<" ";
        }
        std::cout << std::endl;
    }
    std::cout << "*****************************"<< std::endl;


    //创建上下文
    context=CreatContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }
 
    //创建命令队列
    commandQueue=CreateCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }
    
    //创建程序对象
    program=CreateProgram(context, device, "convolve.cl");
    if (program == NULL)
    {
        std::cerr << "Failed to create program" << std::endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }
    
    //创建OpenCL核
    kernel=clCreateKernel(program, "convolve", nullptr);
    if (kernel == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }
    
    //不同
 
    //创建内存对象
    if (!CreateMemObjects(context, memObjects, inputSignal, mask))
    {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }
    
    //设置内核参数
    errNum=clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum|=clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum|=clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    errNum|=clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
    errNum|=clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments." << std::endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }
 
    size_t globalWorkSize[1] = { inputSignalWidth * inputSignalHeight };
    size_t localWorkSize[1] = { 1 };
    
    //为将在设备上执行的内核排队
    errNum=clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }
    
    //执行内核并读出数据
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
                                 0, outputSignalHeight * outputSignalWidth * sizeof(cl_int), outputSignal,
                                 0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return 1;
    }
 
    
    //打开一个文件以写入数据
    std::ofstream outfile3("result.txt");
    //检查文件是否成功打开
    if (!outfile3) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    //保存结果到文件中
    for (int i = 0; i < outputSignalHeight; i++)
    {
        for (int j=0; j<outputSignalWidth; j++) {
            outfile3 << outputSignal[i][j];
            if (j != outputSignalWidth - 1) {
                outfile3 << ",";
            }
        }
        outfile3 << std::endl;
    }
    //关闭文件
    outfile3.close();
    std::cout << "数组已成功保存到文件result.txt" << std::endl;
    
    //打印结果
    for (int i = 0; i < outputSignalHeight; i++)
    {
        for (int j = 0; j < outputSignalWidth; j++) {
            std::cout << outputSignal[i][j] << " ";
        }
        std::cout<<std::endl;
    }
    std::cout << std::endl;
    std::cout << "*****************************" << std::endl;
    std::cout << "Executed program succesfully." << std::endl;

    

    Cleanup(context, commandQueue, program, kernel, memObjects);
    
    return 0;
}

cl_context CreatContext(){
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformID;
    cl_context context=nullptr;
    
    /*首先，选择要运行的OpenCL平台。 对于这个例子，我们
      只需选择第一个可用平台。通常，我们也可以
      查询所有可用平台，然后选择最合适的平台。
    */
    errNum=clGetPlatformIDs(1, &firstPlatformID, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }
    
    /*接下来，在平台上创建一个OpenCL上下文。 尝试
      创建基于GPU的上下文，如果失败，请尝试创建
      基于CPU的上下文。
    */
    
    //创建上下文需要的资源 属性 属性值 0结束
    cl_context_properties contextProperties[]={
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformID,
        0
    };
    
    context=clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, nullptr, nullptr, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }
 
    return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;
 
    // 获取设备缓存的尺寸
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }
 
    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }
 
    // 给设备缓存分配空间
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }
 
    // 选择第一个设备和上下文，创建出一个命令队列
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }
 
    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;
 
    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }
 
    std::ostringstream oss;
    oss << kernelFile.rdbuf();//oss输出kernelFile指向的流缓冲
 
    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    
    //在context上下文上创建程序对象(字符串个数为1)
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }
 
    //编译内核源码
    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // 输出编译错误信息
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);
 
        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);//释放程序对象空间
        return NULL;
    }
 
    return program;
}


bool CreateMemObjects(cl_context context, cl_mem memObjects[3], cl_int (*a)[inputSignalWidth], cl_int (*b)[maskWidth])
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(cl_int) * inputSignalHeight * inputSignalWidth,
                                   a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(cl_int) * maskHeight * maskWidth,
                                   b, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(cl_float) * outputSignalHeight * outputSignalWidth, NULL, NULL);
 
    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }
 
    return true;
}
 
//释放OpenCL资源
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);
 
    if (kernel != 0)
        clReleaseKernel(kernel);
 
    if (program != 0)
        clReleaseProgram(program);
 
    if (context != 0)
        clReleaseContext(context);
 
}


