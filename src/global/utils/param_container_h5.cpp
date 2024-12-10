#if defined(OUTPUT_ENABLED)
#include "utils/param_container_h5.h"

#include "enums.h"
#include "global.h"

#include <H5Cpp.h>

#include <any>
#include <functional>
#include <map>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

// 与原代码类似的trait，判断类型是否有to_string()
namespace prm {
  template <typename T, typename = void>
  struct has_to_string : std::false_type {};

  template <typename T>
  struct has_to_string<T, std::void_t<decltype(std::declval<T>().to_string())>>
    : std::true_type {};

  // 对标量类型的HDF5属性写入帮助函数
  template <typename T>
  void write_scalar(H5::H5Object& obj, const std::string& name, const T& value) {
    H5::DataSpace scalar_space(H5S_SCALAR);
    // 对bool特殊处理，使用unsigned char表示
    if constexpr (std::is_same_v<T, bool>) {
      unsigned char val = value ? 1 : 0;
      auto attr = obj.createAttribute(name, H5::PredType::NATIVE_UCHAR, scalar_space);
      attr.write(H5::PredType::NATIVE_UCHAR, &val);
    } else if constexpr (std::is_same_v<T, Dimension>) {
      // 将Dimension转为unsigned short存储
      unsigned short val = static_cast<unsigned short>(value);
      auto attr = obj.createAttribute(name, H5::PredType::NATIVE_USHORT, scalar_space);
      attr.write(H5::PredType::NATIVE_USHORT, &val);
    } else {
      // 其他类型根据类型选择HDF5原生类型
      // 简单起见，以double,int,unsigned int等类型为例，需要一个type selector
      // 在此简化处理，如下定义
      auto getPredType = []() {
        if constexpr (std::is_same_v<T, double>) return H5::PredType::NATIVE_DOUBLE;
        else if constexpr (std::is_same_v<T, float>) return H5::PredType::NATIVE_FLOAT;
        else if constexpr (std::is_same_v<T, int>) return H5::PredType::NATIVE_INT;
        else if constexpr (std::is_same_v<T, unsigned int>) return H5::PredType::NATIVE_UINT;
        else if constexpr (std::is_same_v<T, std::size_t>) return H5::PredType::NATIVE_ULLONG;
        else if constexpr (std::is_same_v<T, short>) return H5::PredType::NATIVE_SHORT;
        else if constexpr (std::is_same_v<T, unsigned short>) return H5::PredType::NATIVE_USHORT;
        else if constexpr (std::is_same_v<T, long int>) return H5::PredType::NATIVE_LONG;
        else if constexpr (std::is_same_v<T, long double>) return H5::PredType::NATIVE_LDOUBLE;
        else if constexpr (std::is_same_v<T, unsigned long int>) return H5::PredType::NATIVE_ULONG;
        else {
          // 对有to_string的类型存为string
          if constexpr (has_to_string<T>::value) {
            // 有to_string的方法则转为string存储
            // 转为string:
            std::string str_val = value.to_string();
            H5::StrType str_type(0, H5T_VARIABLE);
            auto attr = obj.createAttribute(name, str_type, scalar_space);
            attr.write(str_type, str_val);
            return H5::PredType(); // 返回空，因为已处理
          } else if constexpr (std::is_same_v<T, std::string>) {
            H5::StrType str_type(0, H5T_VARIABLE);
            auto attr = obj.createAttribute(name, str_type, scalar_space);
            attr.write(str_type, value);
            return H5::PredType();
          } else {
            static_assert(!sizeof(T*), "Unsupported scalar type");
          }
        }
      };

      if constexpr (!has_to_string<T>::value && !std::is_same_v<T, std::string>) {
        auto ptype = getPredType();
        auto attr = obj.createAttribute(name, ptype, scalar_space);
        attr.write(ptype, &value);
      }
      // 如果has_to_string<T>::value为true或T=std::string，已经在上面处理过了，不执行此处
    }
  }

  // 写字符串标量
  inline void write_scalar(H5::H5Object& obj, const std::string& name, const std::string& value) {
    H5::DataSpace scalar_space(H5S_SCALAR);
    H5::StrType str_type(0, H5T_VARIABLE);
    auto attr = obj.createAttribute(name, str_type, scalar_space);
    attr.write(str_type, value);
  }

  // 对有to_string类型的标量
  template <typename T>
  void write_scalar_with_tostring(H5::H5Object& obj, const std::string& name, const T& value) {
    std::string val_str = value.to_string();
    write_scalar(obj, name, val_str);
  }

  // 写向量（无需区分has_to_string与否，在写时判断）
  template <typename T>
  void write_vector(H5::H5Object& obj, const std::string& name, const std::vector<T>& vec) {
    // 如果有to_string，则存为字符串数组
    if constexpr (has_to_string<T>::value) {
      std::vector<std::string> str_vec;
      for (auto& v : vec) {
        str_vec.push_back(v.to_string());
      }
      // 写字符串数组
      hsize_t dims[1] = { str_vec.size() };
      H5::DataSpace space(1, dims);
      H5::StrType str_type(0, H5T_VARIABLE);
      auto attr = obj.createAttribute(name, str_type, space);

      std::vector<const char*> cstrs(str_vec.size());
      for (size_t i = 0; i < str_vec.size(); ++i) cstrs[i] = str_vec[i].c_str();
      attr.write(str_type, cstrs.data());
    } else if constexpr (std::is_same_v<T, std::string>) {
      // 字符串数组
      hsize_t dims[1] = { vec.size() };
      H5::DataSpace space(1, dims);
      H5::StrType str_type(0, H5T_VARIABLE);
      auto attr = obj.createAttribute(name, str_type, space);

      std::vector<const char*> cstrs(vec.size());
      for (size_t i = 0; i < vec.size(); ++i) cstrs[i] = vec[i].c_str();
      attr.write(str_type, cstrs.data());
    } else if constexpr (std::is_same_v<T, bool>) {
      // bool数组存为uchar
      std::vector<unsigned char> bool_data(vec.size());
      for (size_t i = 0; i < vec.size(); ++i) bool_data[i] = vec[i] ? 1 : 0;
      hsize_t dims[1] = { bool_data.size() };
      H5::DataSpace space(1, dims);
      auto attr = obj.createAttribute(name, H5::PredType::NATIVE_UCHAR, space);
      attr.write(H5::PredType::NATIVE_UCHAR, bool_data.data());
    } else {
      // 标量类型数组
      // 根据T类型选择HDF5类型(参考write_scalar逻辑)
      auto getPredType = []() {
        if constexpr (std::is_same_v<T, double>) return H5::PredType::NATIVE_DOUBLE;
        else if constexpr (std::is_same_v<T, float>) return H5::PredType::NATIVE_FLOAT;
        else if constexpr (std::is_same_v<T, int>) return H5::PredType::NATIVE_INT;
        else if constexpr (std::is_same_v<T, unsigned int>) return H5::PredType::NATIVE_UINT;
        else if constexpr (std::is_same_v<T, std::size_t>) return H5::PredType::NATIVE_ULLONG;
        else if constexpr (std::is_same_v<T, short>) return H5::PredType::NATIVE_SHORT;
        else if constexpr (std::is_same_v<T, unsigned short>) return H5::PredType::NATIVE_USHORT;
        else if constexpr (std::is_same_v<T, long int>) return H5::PredType::NATIVE_LONG;
        else if constexpr (std::is_same_v<T, long double>) return H5::PredType::NATIVE_LDOUBLE;
        else if constexpr (std::is_same_v<T, unsigned long int>) return H5::PredType::NATIVE_ULONG;
        else {
          static_assert(!sizeof(T*), "Unsupported vector element type");
        }
      };

      auto ptype = getPredType();
      hsize_t dims[1] = { vec.size() };
      H5::DataSpace space(1, dims);
      auto attr = obj.createAttribute(name, ptype, space);
      attr.write(ptype, vec.data());
    }
  }

  // 写 pair<T,T>
  template <typename T>
  void write_pair(H5::H5Object& obj, const std::string& name, const std::pair<T, T>& p) {
    // 将pair转为vector处理
    std::vector<T> vec = { p.first, p.second };
    write_vector(obj, name, vec);
  }

  // 写 vector<pair<T,T>>
  template <typename T>
  void write_vec_pair(H5::H5Object& obj, const std::string& name, const std::vector<std::pair<T,T>>& vp) {
    // 展开为一个扁平化的vector
    std::vector<T> flat;
    flat.reserve(vp.size()*2);
    for (auto& p : vp) {
      flat.push_back(p.first);
      flat.push_back(p.second);
    }
    write_vector(obj, name, flat);
  }

  // 写 vector<vector<T>>
  template <typename T>
  void write_vec_vec(H5::H5Object& obj, const std::string& name, const std::vector<std::vector<T>>& vvt) {
    // 扁平化为单一vector
    std::vector<T> flat;
    for (auto& v : vvt) {
      flat.insert(flat.end(), v.begin(), v.end());
    }
    write_vector(obj, name, flat);
  }

  // 写 map<string,T>
  template <typename T>
  void write_dict(H5::H5Object& obj, const std::string& name, const std::map<std::string,T>& dict) {
    // 将字典的每个键值对写为单独的属性，名称为 name_key
    for (auto& [k,v] : dict) {
      if constexpr (has_to_string<T>::value) {
        std::string val_str = v.to_string();
        write_scalar(obj, name + "_" + k, val_str);
      } else {
        // 使用write_scalar或write_vector取决于v类型是否为标量或容器
        // 简化处理：假设都是标量
        write_scalar(obj, name + "_" + k, v);
      }
    }
  }

  // 通用写函数，根据类型选择合适函数调用
  // 标量类型：直接write_scalar
  // 有to_string的标量：write_scalar_with_tostring
  // pair, vector, vector<pair>, vector<vector>, dict 使用上面定义的辅助函数

  // 简化逻辑：根据type判断结构，类型已在原代码中区分过，这里沿用类似方案

  static std::map<std::type_index, std::function<void(H5::H5Object&, const std::string&, std::any)>> write_functions;

  template <typename T>
  void register_write_function() {
    write_functions[std::type_index(typeid(T))] =
      [](H5::H5Object& obj, const std::string& name, std::any a) {
        T val = std::any_cast<T>(a);
        if constexpr (has_to_string<T>::value) {
          write_scalar_with_tostring(obj, name, val);
        } else {
          write_scalar(obj, name, val);
        }
      };
  }

  template <typename T>
  void register_write_function_for_pair() {
    write_functions[std::type_index(typeid(std::pair<T, T>))] =
      [](H5::H5Object& obj, const std::string& name, std::any a) {
        write_pair(obj, name, std::any_cast<std::pair<T, T>>(a));
      };
  }

  template <typename T>
  void register_write_function_for_vector() {
    write_functions[std::type_index(typeid(std::vector<T>))] =
      [](H5::H5Object& obj, const std::string& name, std::any a) {
        write_vector(obj, name, std::any_cast<std::vector<T>>(a));
      };
  }

  template <typename T>
  void register_write_function_for_vector_of_pair() {
    write_functions[std::type_index(typeid(std::vector<std::pair<T, T>>))] =
      [](H5::H5Object& obj, const std::string& name, std::any a) {
        write_vec_pair(obj, name, std::any_cast<std::vector<std::pair<T, T>>>(a));
      };
  }

  template <typename T>
  void register_write_function_for_vector_of_vector() {
    write_functions[std::type_index(typeid(std::vector<std::vector<T>>))] =
      [](H5::H5Object& obj, const std::string& name, std::any a) {
        write_vec_vec(obj, name, std::any_cast<std::vector<std::vector<T>>>(a));
      };
  }

  template <typename T>
  void register_write_function_for_dict() {
    write_functions[std::type_index(typeid(std::map<std::string, T>))] =
      [](H5::H5Object& obj, const std::string& name, std::any a) {
        write_dict(obj, name, std::any_cast<std::map<std::string,T>>(a));
      };
  }

  void write_any(H5::H5Object& obj, const std::string& name, std::any a) {
    auto it = write_functions.find(std::type_index(a.type()));
    if (it != write_functions.end()) {
      it->second(obj, name, a);
    } else {
      throw std::runtime_error("No write function registered for this type");
    }
  }

  // 在此实现Parameters::write方法，但使用HDF5对象代替ADIOS2::IO
  // 在实际中，您需要在param_container.h中将write(...)声明为使用H5::H5Object&参数
  void Parameters::write(H5::H5Object& obj) const {
    // 在这里注册所有类型与写函数的映射（与原代码类似）
    // 您可以根据需要减少或增加类型支持

    register_write_function<double>();
    register_write_function<float>();
    register_write_function<int>();
    register_write_function<std::size_t>();
    register_write_function<unsigned int>();
    register_write_function<long int>();
    register_write_function<long double>();
    register_write_function<unsigned long int>();
    register_write_function<short>();
    register_write_function<bool>();
    register_write_function<unsigned short>();
    register_write_function<std::string>();
    register_write_function<Dimension>();
    // 其他ntt命名空间类型也注册，如下（假设有to_string或可作为标量处理）
    register_write_function<ntt::FldsBC>();
    register_write_function<ntt::PrtlBC>();
    register_write_function<ntt::Coord>();
    register_write_function<ntt::Metric>();
    register_write_function<ntt::SimEngine>();
    register_write_function<ntt::PrtlPusher>();

    // 注册pair
    #define REG_PAIR(T) register_write_function_for_pair<T>()
    REG_PAIR(double);REG_PAIR(float);REG_PAIR(int);REG_PAIR(std::size_t);
    REG_PAIR(unsigned int);REG_PAIR(long int);REG_PAIR(long double);
    REG_PAIR(unsigned long int);REG_PAIR(short);REG_PAIR(unsigned short);
    REG_PAIR(std::string);REG_PAIR(ntt::FldsBC);REG_PAIR(ntt::PrtlBC);
    REG_PAIR(ntt::Coord);REG_PAIR(ntt::Metric);REG_PAIR(ntt::SimEngine);
    REG_PAIR(ntt::PrtlPusher);

    // 注册vector
    #define REG_VEC(T) register_write_function_for_vector<T>()
    REG_VEC(double);REG_VEC(float);REG_VEC(int);REG_VEC(std::size_t);
    REG_VEC(unsigned int);REG_VEC(long int);REG_VEC(long double);
    REG_VEC(unsigned long int);REG_VEC(short);REG_VEC(unsigned short);
    REG_VEC(std::string);REG_VEC(ntt::FldsBC);REG_VEC(ntt::PrtlBC);
    REG_VEC(ntt::Coord);REG_VEC(ntt::Metric);REG_VEC(ntt::SimEngine);
    REG_VEC(ntt::PrtlPusher);

    // 注册vector<pair<T,T>>
    #define REG_VEC_PAIR(T) register_write_function_for_vector_of_pair<T>()
    REG_VEC_PAIR(double);REG_VEC_PAIR(float);REG_VEC_PAIR(int);REG_VEC_PAIR(std::size_t);
    REG_VEC_PAIR(unsigned int);REG_VEC_PAIR(long int);REG_VEC_PAIR(long double);
    REG_VEC_PAIR(unsigned long int);REG_VEC_PAIR(short);REG_VEC_PAIR(unsigned short);
    REG_VEC_PAIR(std::string);REG_VEC_PAIR(ntt::FldsBC);REG_VEC_PAIR(ntt::PrtlBC);
    REG_VEC_PAIR(ntt::Coord);REG_VEC_PAIR(ntt::Metric);REG_VEC_PAIR(ntt::SimEngine);
    REG_VEC_PAIR(ntt::PrtlPusher);

    // 注册vector<vector<T>>
    #define REG_VEC_VEC(T) register_write_function_for_vector_of_vector<T>()
    REG_VEC_VEC(double);REG_VEC_VEC(float);REG_VEC_VEC(int);REG_VEC_VEC(std::size_t);
    REG_VEC_VEC(unsigned int);REG_VEC_VEC(long int);REG_VEC_VEC(long double);
    REG_VEC_VEC(unsigned long int);REG_VEC_VEC(short);REG_VEC_VEC(unsigned short);
    REG_VEC_VEC(std::string);REG_VEC_VEC(ntt::FldsBC);REG_VEC_VEC(ntt::PrtlBC);
    REG_VEC_VEC(ntt::Coord);REG_VEC_VEC(ntt::Metric);REG_VEC_VEC(ntt::SimEngine);
    REG_VEC_VEC(ntt::PrtlPusher);

    // 注册dict
    #define REG_DICT(T) register_write_function_for_dict<T>()
    REG_DICT(double);REG_DICT(float);REG_DICT(int);REG_DICT(std::size_t);
    REG_DICT(unsigned int);REG_DICT(long int);REG_DICT(long double);
    REG_DICT(unsigned long int);REG_DICT(short);REG_DICT(unsigned short);
    REG_DICT(std::string);
    // 对有to_string的类型字典写法已在write_dict中处理，这里略

    // 最终写出所有vars
    for (auto& [key, value] : allVars()) {
      try {
        write_any(obj, key, value);
      } catch (const std::exception& e) {
        // 继续下一个key
        continue;
      }
    }
  }

} // namespace prm
#endif
