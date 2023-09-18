#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <OpenCL/opencl.h>

#include "instance.h"

namespace TYPES {
          template <typename type>
          struct VALID : std::false_type{};

          template <>
          struct VALID<int> : std::false_type{};
          template <>
          struct VALID<float> : std::false_type{};
          template <>
          struct VALID<double> : std::false_type{};

          template<typename type>
          const bool valid = VALID<type>::value;
}

// 
template <typename type, std::enable_if<TYPES::valid<type>, int> = 0>
class zmatric;

template <typename type, std::enable_if<TYPES::valid<type>, int> = 0>
class data_type : OpenCL_base {
          type* data;
public:
          typedef type TYPE;
          data_type(type* data);
          ~data_type();

          type* get_data();
};

/**
 * Main function is to do calculation.
 * 
 */
class data_base : public OpenCL_base {
          const OpenCL *cl;
public:
          explicit data_base(const OpenCL *cl): cl(cl) {}
          ~data_base() {}
          /**
           * 单一计算：运行计算需要kernel 和 command queue
           * 在device上开辟buffer，将数据传入device，如果数据量过大则分批次计算，最后进行归并
           * 通过kernel_name得到kernel function，将input_row, intpu_col, intpu, output绑定到kernel function的输入位置
           * 通过outpu_row, outpu_col可以通过output还原为zmatric格式的结果
           * 
           * 需要将结果从device中取回内存吗？
           * 运算调用包括：
           *        Matric x Matric
           *        Vector x Vector (row x col & col x row)
           *        Value  x Value
           *        Matric x Vector
           *        Vector x Matric
           *        Matric x Value
           *        Value  x Matric
           * 需要对kernel_name的正确性做一次检测，或者在上层调用做检测
           */ 
          template <typename type, std::enable_if<TYPES::valid<type>, int> = 0>
          uint32_t caculate (
                    std::string &kernel_name, 
                    // size_t input_row1, size_t intpu_col1, type *input1, 
                    // size_t input_row2, size_t intpu_col2, type *input2, 
                    // size_t output_row, size_t output_col, type *output
                    zmatric<type>* input1, zmatric<type>* input2, zmatric<type>* output
          ) const {

          }
};

/**
 * Implimented by data_base
 * 
 */ 
template <typename type, std::enable_if<TYPES::valid<type>, int>>
class zmatric /*: public data_base*/ : OpenCL_base {
          type **data;
          static std::unordered_map<type **, size_t> reference;       // Reference of data
          /**
           * col_size == 1 && row_size > 1时为列向量
           * row_size == 1 && col_size > 1时为行向量
           * row_size == col_size == 1时为一个数值变量
           */
          size_t col_size, row_size;
          static const data_base* db;

          bool full_rank, normalized, symmetry, orthognal;

public:
          typedef type type_category;
          enum ztype { MATRIC, VECTOR, VALUE } t;

          explicit zmatric(const OpenCL *cl, const std::vector<std::vector<type>> &source)
          : /*data_base(cl),*/ full_rank(false), normalized(false), symmetry(false), orthognal(false) {

                    if (!db) db = new data_base(cl);

                    row_size = source.size();
                    if (!row_size) {
                              // fprintf(stdout, "Invalid matric source!");
                              throw "Invalid matric source!";
                    }
                    col_size = source[0].size();
                    for (int i = 1; i < row_size; ++i) 
                    if (col_size == source[i].size()) continue;
                    else {
                              // fprintf(stdout, "Invalid matric source!");
                              // std::abort();
                              throw "Invalid matric source!";
                    }

                    data = (type **)malloc(sizeof(type *) * row_size);
                    for (int i = 0; i < row_size; ++i) {
                              data[i] = (type *)malloc(sizeof(type) * col_size);
                              // memset(data[i], 0, sizeof(type) * col_size);
                              for (int j = 0; j < col_size; ++j) data[i][j] = source[i][j];
                    }

                    reference[data] = 1;
          };
          ~zmatric() {
                    if (reference[data] == 1) {
                              for (int i = 0; i < row_size; ++i)
                              if (data[i] != nullptr) free(data[i]);
                              free(data);

                              reference[data] = 0;
                              // Erase from reference
                              reference.erase(data);
                    }

                    reference[data]--;
          };

          zmatric(const zmatric &zm)
          : data(zm.data), col_size(zm.col_size), row_size(zm.row_size), 
          full_rank(zm.full_rank), normalized(zm.normalized), symmetry(zm.symmetry), orthognal(zm.orthognal) 
          { reference[data]++; };

          uint32_t operator=(const zmatric &zm) {
                    
          };

          const zmatric transpos() const {};
          const zmatric inverse() const;
          const zmatric normalize() const;
          const zmatric orthognalize() const;

          std::vector<zmatric> split_into_eigen_spaces() const;
          bool split_into_eigenvectors(zmatric &eigenvalues, zmatric &eigenvectors) const;
          bool svd(zmatric &v, zmatric &u, zmatric &t) const;

          bool is_full_rank() const;
          bool is_normalized() const;
          bool is_symmetry() const;
          bool is_orthognal() const;

          const zmatric operator+(const zmatric &zm) const;
          template <typename T, std::enable_if<TYPES::valid<T>, int> = 0>
          const zmatric operator+(const T &val) const;

          const zmatric operator-(const zmatric &zm) const;
          template <typename T, std::enable_if<TYPES::valid<T>, int> = 0>
          const zmatric operator-(const T &val) const;

          const zmatric dot(const zmatric &zm) const;

          friend class zmatric;
};

// Used to construct data structure and these constructed can do caculation through current OpenCL instance
void* create_data_structure(const OpenCL &cl);