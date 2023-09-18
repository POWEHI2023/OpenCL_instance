#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <OpenCL/opencl.h>

class error {
          std::string msg;
public:
          error (): msg("") {};
          ~error () {};

          // Rewrite copy assigments for simplifying the usage of error
          error(const error &e);
          error(const std::string &str);
          uint32_t operator=(const std::string &str) {
                    msg = str;
                    return 0;
          };
          uint32_t operator=(const error &e);

          // Process condition judgements like e == nullptr / NULL; or e != nullptr / NULL;
          bool operator==(int);
          bool operator==(std::nullptr_t);
          bool operator!=(int);
          bool operator!=(std::nullptr_t);

          bool operator+(std::string &str);

          // Display what the error message is when error is not empty
          const char * what () { return msg.c_str(); };
};

class runnable {
public:

};

uint32_t call_system (const char *cmd, std::string &res) {
          FILE *fp = NULL;
          // 10240 = 40 x 256: This means at most 40 kernel file which with suffix '.cl'
          char result[10240] = {0}, buf[256] = {0};

          if ( (fp = popen(cmd, "r")) == NULL) {
                    fprintf(stdout, "Popen error!\n");
                    return -1;
          }

          while (fgets(buf, sizeof(buf), fp)) strcat(result, buf);
          pclose(fp);
          res = result;

          return 0;
}
// システムコール ! (中二 污染源-_-)
typedef uint32_t(* system_call_t)(const char*, std::string &);
system_call_t system_call = call_system;

struct base_category {};
struct body_category {};
struct matrix_category {};
struct vector_category {};
struct variable_category {};

/**
 * OpenCL的基础部分，用于后续所有关于OpenCL类型来继承
 * 包含运行OpenCL的数据所拥有的公共特征
 */
class OpenCL_base {

public:
          // typedef base_category inner_category;
};

class data_base;
/**
 * OpenCL的实体部分，承担OpenCL的初始化工作，以及维持基本的数据用来运行OpenCL计算
 * 后续通过OpenCL创建用于计算的数据结构，数组/向量/变量
 * 计算通过重写数据结构的operator实现，对于非OpenCL数据需要有具体处理
 * 
 * 为了减少运行时开销，虽然不做声明，默认所有函数throw()，即任何异常需要在类内被处理消化
 * 为什么不强制声明throw(): 在声明throw()后即使没有任何异常被抛出，开销还是会大于不添加任何声明的情况
 * 并且即使做出强制不抛出任何异常的声明，却无法保证不调用任何有同样约束的函数
 */
class OpenCL : public OpenCL_base {
public:
          class backsteper {
                    std::vector< std::function<void()> > funcs;
          public:
                    template <typename F>
                    void append(F const &f) 
                    { funcs.push_back(f); };
                    void clear() 
                    { funcs.clear(); };
                    void recycle() { 
                              for (auto each : funcs) each();
                              funcs.clear();
                    }
          };

private:
          cl_platform_id platform;                // Compute platform
          cl_device_id *device_id;                // Compute device ids
          cl_uint device_number;                  // Device number
          cl_context context;                     // Compute context
          cl_command_queue *commands;             // Compute command queue

          static const std::string program_path;  // Program path

          std::vector<cl_program> programs;       // Compute programs
          std::vector<cl_kernel> kernels;         // Compute kernels
          std::vector<uint32_t> kernel_counts;    // Kernel count for each program

          /**
           * Device and it's Command has same index
           * Record device & program pair, program & kernel function index
           */
          std::vector<int> dp_pair;               // Program and binded device
          std::map<int, int> pk_ref;              // Program and main kernel function
          std::map<std::string, int> nk_ref;      // Function name and it's kernel function

          static bool main_body;
          class accumulator {
                    size_t value;
          public:
                    accumulator(): value(-1) {}
                    ~accumulator() {}

                    void set (size_t v) { value = v; }
                    size_t mget (size_t m) {
                              if (value == std::numeric_limits<size_t>::max()) value = 0;
                              else value += 1;
                              return value < m ? value : (value = 0);
                    }
                    // 缺少回滚功能
                    void roll_back();
          };
          accumulator acc;                        // Counter for create program
          backsteper bs;

          uint32_t load_platform(int platform_id, error &e) {
                    size_t num = OpenCL::platform_number(e);
                    if (num <= 0) {
                              e = "Fail to get any platform!";
                              return -1;
                    }
                    if (platform_id <= 0 || platform_id > num) {
                              e = "Invalid platform id!";
                              return -1;
                    }

                    cl_int ret = clGetPlatformIDs(platform_id, &platform, NULL);
                    if (ret != CL_SUCCESS) {
                              e = "Load platform error!";
                              return -1;
                    }
                    return 0;
          }

          uint32_t load_devices(const cl_device_type type, error &e) {
                    cl_int ret = clGetDeviceIDs(platform, type, 0, NULL, &device_number);
                    if (ret != CL_SUCCESS) {
                              e = "Get device number error!";
                              return -1;
                    } // device_number = 1;
                    device_id = (cl_device_id *)malloc(sizeof(cl_device_id) * device_number);
                    if (!device_id) {
                              e = "Allocate device memory error, device number " + std::to_string(device_number) + "!";
                              return -1;
                    }
                    ret = clGetDeviceIDs(platform, type, device_number, device_id, NULL);
                    if (ret != CL_SUCCESS) {
                              e = "Get devices error!";
                              return -1;
                    }
                    fprintf(stdout, "LOAD: Find %d devices.\n", device_number);
                    return 0;
          }

          uint32_t load_context(error &e) {
                    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
                    cl_int ret;
                    context = clCreateContext(properties, device_number, device_id, NULL, NULL, &ret);
                    if (ret != CL_SUCCESS) {
                              e = "Create context error!";
                              return -1;
                    }
                    return 0;
          }

          uint32_t load_commands(error &e) {
                    commands = (cl_command_queue *)malloc(sizeof(cl_command_queue) * device_number);
                    if (commands == nullptr) {
                              e = "Allocate for commands error!";
                              return -1;
                    }

                    cl_int ret;
                    for (int i = 0; i < device_number; ++i) {
                              commands[i] = clCreateCommandQueue(context, device_id[i], 0, &ret);
                              if (ret != CL_SUCCESS) {
                                        e = "Create command queue error!";
                                        return -1;
                              }
                    }
                    return 0;
          }

          // Type to distinguish different program
          uint32_t load_program(const std::string &type, error &e) {
                    // Get program file number
                    std::string output, cmd = "ls -Ap " + program_path + type + "/*.cl | grep -v /$ | wc -l";           // std::cout << cmd << std::endl;
                    fprintf(stdout, "Check directory %s ...\n", (program_path + type).c_str());

                    uint32_t ret = call_system(cmd.c_str(), output);                                                    // std::cout << output << std::endl;
                    if (ret == -1) {
                              e = "System call make some problems!";
                              return -1;
                    }
                    int file_number = atoi(output.c_str());
                    if (file_number == 0) {
                              return 0;
                    }
                    // 后面用日志模块替代直接打印的语句
                    if (file_number > 1) fprintf(stdout, "Find there are %d files exist in the directory.\n", file_number);
                    else fprintf(stdout, "Find there is 1 file exists in the directory.\n");

                    // Get program file name
                    std::vector<std::string> file_name(file_number, "");
                    cmd = "ls " + program_path + type + "/*.cl";                // std::cout << cmd << std::endl;
                    ret = call_system(cmd.c_str(), output);                     // std::cout << output << std::endl;
                    if (ret == -1) {
                              e = "System call meets some problems!";
                              return -1;
                    }
                    for (int i = 0, ptr = 0; i < output.size() && ptr < file_number; ++i) {
                              if (output[i] == ' ') continue;
                              if (output[i] == '\n') {
                                        ptr++; continue;
                              }
                              file_name[ptr] += output[i];
                    }

                    for (int i = 0; i < file_number; ++i) fprintf(stdout, "FILE: %s\n", file_name[i].c_str());

                    // Now we have files' name, then we need transform file to string for loading into device as program
                    FILE *program_handle;
                    std::vector<std::string> program_buffer(file_number, "");
                    // std::vector<size_t> program_size(file_number);
                    size_t program_size[file_number];
                    for (int i = 0; i < file_number; ++i) {
                              program_handle = fopen(file_name[i].c_str(), "r");
                              fseek(program_handle, 0, SEEK_END);
                              program_size[i] = ftell(program_handle);
                              rewind(program_handle);
                              char buffer[program_size[i]];
                              fread(buffer, sizeof(char), program_size[i], program_handle);
                              program_buffer[i] = buffer;
                              fclose(program_handle);
                    }

                    char *program_buffer_[file_number];
                    for (int i = 0; i < file_number; ++i) 
                    program_buffer_[i] = const_cast<char *>(program_buffer[i].c_str());
                    cl_int r;

                    // Multiple program files in the directory, mixed into one program
                    cl_program program = clCreateProgramWithSource(context, file_number, const_cast<const char **>(program_buffer_), program_size, &r);
                    if (r != CL_SUCCESS) {
                              e = "Create program with source error!";
                              return -1;
                    }
                    programs.push_back(program);
                    bs.append([=](){ 
                              if (programs.back() != 0) clReleaseProgram(programs.back());
                              programs.pop_back();
                    });
                    /**
                     * 可以添加代码：将首次添加的代码编译后存储为二进制文件.bin，由于OpenCL是动态加载的核函数
                     * 首次编译后再加载同样的运算时可以减小开销
                     * 
                     * 构造时可以减少加载的运算，只加载常用必要的一些运算，可以通过额外的功能模块记录运算的频率，作为加载时参考
                     * 附加模块可以绑定动作，在调用时触发必要并行处理，调用线程池模块添加并行任务
                     */
                    // Build program, note: One program just involve one essential caculation, so we assigment one device for one caculation
                    int ptr_device = acc.mget(device_number);
                    fprintf(stdout, "Create program(%ld) in device %d.\n", programs.size() - 1, ptr_device);
                    ret = clBuildProgram(program, 1, &device_id[ptr_device], NULL, NULL, NULL);
                    if (ret != CL_SUCCESS) {
                              size_t len;
                              char buffer[2048];
                              e = "Error: Failed to build program executable!";
                              // printf("Error: Failed to build program executable!\n");
                              clGetProgramBuildInfo(program, device_id[ptr_device], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                              e = buffer;
                              // programs.pop_back();
                              bs.recycle();
                              // exit(1);
                              return -1;
                    }
                    // Record the device which process the porgram
                    dp_pair.push_back(ptr_device);
                    bs.append([=]() { dp_pair.pop_back(); });
                    // display_program(e);
                    return load_kernels(type, e);
          };

          /**
           * LoadProgram返回错误时，特别是在字符串中建立Program时返回错误时
           * 应该不影响正常运行的部分参数
           */ 
          uint32_t load_program_from_chars (const char *source, error &e) {
                    cl_int err;
                    cl_program program = clCreateProgramWithSource(context, 1, (const char **) & source, NULL, &err);
                    if (err != CL_SUCCESS || !program) {
                              e = "Failed to create compute program!";
                              return -1;
                    }
                    
                    programs.push_back(program);

                    int ptr_device = acc.mget(device_number);
                    fprintf(stdout, "Create program from source in device %d.\n", ptr_device);
                    err = clBuildProgram(program, 1, &device_id[ptr_device], NULL, NULL, NULL);
                    if (err != CL_SUCCESS) {
                              size_t len;
                              char buffer[2048];
                              e = "Error: Failed to build program executable!";
                              // printf("Error: Failed to build program executable!\n");
                              clGetProgramBuildInfo(program, device_id[ptr_device], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                              e = buffer;
                              programs.pop_back();
                              // 需要将ptr_device回退一位吗？
                              // exit(1);
                              return -1;
                    }
                    dp_pair.push_back(ptr_device);

                    return load_kernels("", e);
          }

          /**
           * Program binded with a device, and each program juse one essential kernel function
           * RefelectionCondition: Device 1-1-> Program 1-n-> KernelFunction (One essential)
           * Essential kernel function has a special function name, which name same as it's type name
           * If not, report an error and return a error code -1
           */ 
          uint32_t load_kernels(const std::string &type, error &e) {
                    // std::cout << "Program number: " << programs.size() << std::endl;
                    int index = programs.size() - 1;
                    cl_uint kernel_num;
                    cl_int ret = clCreateKernelsInProgram(programs[index], 0, NULL, &kernel_num);
                    if (ret != CL_SUCCESS) {
                              e = "Get kernel number error!" + std::to_string(ret);
                              // programs.pop_back();
                              bs.recycle();
                              return -1;
                    }
                    cl_kernel *ks = (cl_kernel *)malloc(sizeof(cl_kernel) * kernel_num);
                    if (ks == nullptr) {
                              e = "Alloc for kernels error!";
                              // programs.pop_back();
                              bs.recycle();
                              free(ks);
                              return -1;
                    }
                    ret = clCreateKernelsInProgram(programs[index], kernel_num, ks, NULL);
                    if (ret != CL_SUCCESS) {
                              e = "Create kernels for program error!";
                              // programs.pop_back();
                              bs.recycle();
                              free(ks);
                              return -1;
                    }

                    /*size_t kernel_count;
                    ret = clGetProgramInfo(programs[index], CL_PROGRAM_NUM_KERNELS, sizeof(size_t), &kernel_count, NULL);
                    if (ret != CL_SUCCESS) {
                              e = "Get program info error!";
                              programs.pop_back();
                              free(ks);
                              return -1;
                    }*/
                    kernel_counts.push_back(kernel_num);
                    bs.append([=]() { kernel_counts.pop_back(); });

                    printf("Find %ud kernels!\n", kernel_num);

                    int begin_ = kernels.size();
                    for (int i = 0; i < kernel_num; ++i)
                    kernels.push_back(ks[i]);
                    bs.append([=]() {
                              for (int i = 0; i < kernel_num; ++i) {
                                        if (kernels.back() != 0) clReleaseKernel(kernels.back());
                                        kernels.pop_back();
                              }
                    });
                    free(ks);

                    // Reflece kernels into function name
                    bool matched = false;
                    for (int i = begin_, len = kernels.size(); i < len; ++i) {
                              size_t param_value_size_ret;
                              ret = clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, 0, NULL, &param_value_size_ret);
                              if (ret != CL_SUCCESS) {
                                        e = "Get kernel name size error!";
                                        // release_program(index);
                                        bs.recycle();
                                        return -1;
                              }

                              char *name = (char *)malloc(sizeof(char) * (param_value_size_ret + 1));
                              memset(name, 0, param_value_size_ret + 1);
                              ret = clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, param_value_size_ret, name, NULL);
                              name[param_value_size_ret] = '\0';
                              if (ret != CL_SUCCESS) {
                                        e = "Get kernel name error!";
                                        // release_program(index);
                                        bs.recycle();
                                        return -1;
                              }

                              fprintf(stdout, "Name: %s\n", name);
                              if (strcmp(name, type.c_str()) == 0) {
                                        // Guarantee just one main kernel function for each program
                                        if (matched) {
                                                  e = "Too much main kernel functions!";
                                                  // release_program(index);
                                                  bs.recycle();
                                                  return -1;
                                        }
                                        /**
                                         * How to find command through function namm? 
                                         * dp_pair[ pk_ref[ nk_ref[name] ] ] == device index == command index
                                         */
                                        nk_ref[type] = i;
                                        pk_ref[i] = index;

                                        bs.append([=]() {
                                                  nk_ref.erase(type);
                                                  pk_ref.erase(i);
                                        });
                                        matched = true;
                              } else if (type == "" && kernel_num == 1) {
                                        nk_ref[name] = i;
                                        pk_ref[i] = index;

                                        bs.append([=]() {
                                                  nk_ref.erase(type);
                                                  pk_ref.erase(i);
                                        });
                                        matched = true;
                              }
                              free(name);
                              // check_reflect_status();
                    }
                    
                    if (!matched) {
                              e = "No matched kernel function found in the program!";
                              /*for (auto &&[k, v] : pk_ref) if (v == index) { 
                                        pk_ref.erase(k); 
                                        for (auto &&[i, j] : nk_ref) if (j == k) 
                                        { nk_ref.erase(i); break; }
                                        break;
                              }
                              release_program(index);*/
                              bs.recycle();
                              printf("\n");
                              // check_reflect_status();
                              return -1;
                    }

                    bs.clear();
                    return 0;
          }

          uint32_t release_program(int index) {
                    if (programs.size() <= index) {
                              fprintf(stdout, "Invelid index for releasing program!");
                              return -1;
                    }
                    int kbegin = 0, i = 0;
                    for (; i < index; ++i) kbegin += kernel_counts[i];
                    if (kernels.size() <= kbegin || kernels.size() < kbegin + kernel_counts[i]) {
                              fprintf(stdout, "No enough kernels!");
                              return 1;           // Error
                    }
                    for (int j = kbegin + kernel_counts[i]; j < kernels.size(); ++j, ++kbegin) {
                              if (kernels[kbegin] != 0) clReleaseKernel(kernels[kbegin]);
                              kernels[kbegin] = kernels[j];
                    }
                    while (kbegin < kernels.size()) kernels.pop_back();

                    if (programs[index] != 0) clReleaseProgram(programs[index]);
                    programs.erase(programs.begin() + index);
                    dp_pair.erase(dp_pair.begin() + index);

                    return 0;
          };

          uint32_t recycle(int x) {
                    switch (x) {
                              case 0: {
                                        for (std::vector<cl_kernel>::iterator k_iter = kernels.begin(); k_iter != kernels.end(); ++k_iter)
                                        if (*k_iter != 0) clReleaseKernel(*k_iter);
                              }
                              case 1: {
                                        for (std::vector<cl_program>::iterator p_iter = programs.begin(); p_iter != programs.end(); ++p_iter)
                                        if (*p_iter != 0) clReleaseProgram(*p_iter);
                              }
                              case 2: {
                                        for (int i = 0; i < device_number; ++i)
                                        if (commands[i]) clReleaseCommandQueue(commands[i]);
                                        if (commands != nullptr) free(commands);
                              }
                              case 3: {
                                        clReleaseContext(context);

                                        if (device_id != nullptr) {
                                                  for (int i = 0; i < device_number; ++i)
                                                  if (device_id[i] != 0) clReleaseDevice(device_id[i]);
                                                  free(device_id);
                                        }
                              }
                              case -1: { /*Nothing*/ }
                    }
                    // printf("A OpenCL instance has been released!\n");
                    return 0;
          };

          std::string display_program(error &e) const {
                    size_t res;
                    clGetProgramInfo(programs[0], CL_PROGRAM_KERNEL_NAMES, sizeof(size_t), &res, NULL);
                    std::cout << res << std::endl;

                    return 0;
          }

          void check_reflect_status() const {
                    fprintf(stdout, "ProgramSize: %ld, KernelSize: %ld\n", programs.size(), kernels.size());
                    fprintf(stdout, ": dp_pair: \n");
                    for (int i = 0; i < dp_pair.size(); ++i) fprintf(stdout, "\tProgram:%d => Device:%d\n", i, dp_pair[i]);
                    fprintf(stdout, ": pk_ref: \n");
                    for (auto &&[k, v] : pk_ref) fprintf(stdout, "\tKernelF:%d => Program:%d\n", k, v);
                    fprintf(stdout, ": nk_ref: \n");
                    for (auto &&[k, v]: nk_ref) fprintf(stdout, "\tKernelFName:%s => KernelF:%d\n", k.c_str(), v);
          }


          void init_(int platform_type, error &e) {
                    if (load_platform(platform_type, e) || load_devices(CL_DEVICE_TYPE_GPU, e)) {
                              std::cout << e.what() << std::endl;
                              recycle(-1);
                              std::abort();
                    }
                    if (load_context(e)) {
                              std::cout << e.what() << std::endl;
                              recycle(3);
                              std::abort();
                    }
                    if (load_commands(e)) {
                              std::cout << e.what() << std::endl;
                              recycle(2);
                              std::abort();
                    }
          }
public:
          // typedef body_category inner_category;
          
          OpenCL() {
                    /*if (main_body) {
                              std::abort();
                    }*/
                    // printf("Start construct a OpenCL without any parameters!\n");
                    error e;
                    init_(1, e);

                    // Do we need to forbid constructing multiplu OpenCL instance?
                    // main_body = true;
          };
          ~OpenCL() {
                    // printf("Start release a OpenCL instance!\n");
                    recycle(0);
          };

          explicit OpenCL(int platform_type) {
                    error e;
                    init_(platform_type, e);

                    // The `type` name has been orderd for testing
                    if (load_program("matrix", e) || load_program("vector", e)) {
                              std::cout << e.what() << std::endl;
                              recycle(0);
                              std::abort();
                    }

                    // printf("Create over!\n");
          };

          /**
           * When use copy assigments we need call clRetainXXX() to increase the reference count for each copied constant
           * In case of some copied OpenCL be released, some OpenCL's constants also relsased consistently
           */ 
          OpenCL(const OpenCL &cl);
          uint32_t operator=(const OpenCL &cl);

          // 通过platform number限制platform参数，用户可以通过当前接口构造OpenCL instacne
          static size_t platform_number(error &e) {
                    cl_uint size;
                    cl_int err = clGetPlatformIDs(0, NULL, &size);
                    if (err != CL_SUCCESS) {
                              e = "Get platform number error!";
                              return -1;
                    }
                    return size;
          }

          // 加载成功后可以通过主函数名直接调用，加载失败会推出加载的program并返回错误码-1
          uint32_t load_program_from_source(const char* source, error &e) 
          { return load_program_from_chars(source, e); };

          uint32_t release_program(const std::string &name) {
                    int index = pk_ref[ nk_ref[name] ];
                    pk_ref.erase(nk_ref[name]);
                    nk_ref.erase(name);
                    return release_program(index); 
          }

          friend class data_base;
};

const std::string OpenCL::program_path = "${INSTANCE}/kernel/";
bool OpenCL::main_body = false;