/**
 * @file output/utils/attr_writer.h
 * @brief Functions to write custom type attributes using HDF5
 * @implements
 *   - out::writeAnyAttr -> void
 *   - out::defineAttribute<> -> void
 * @namespaces:
 *   - out::
 */

#ifndef OUTPUT_UTILS_ATTR_WRITER_H
#define OUTPUT_UTILS_ATTR_WRITER_H

#include <H5Cpp.h>

#include <any>
#include <functional>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>
#include <stdexcept>



namespace out {

namespace detail {

    // 针对标量类型的模板函数
    template <typename T>
    struct H5PredTypeSelector {};

    // 根据类型选择合适的HDF5原生类型
    template<> struct H5PredTypeSelector<int> { static H5::PredType type() { return H5::PredType::NATIVE_INT; } };
    template<> struct H5PredTypeSelector<short> { static H5::PredType type() { return H5::PredType::NATIVE_SHORT; } };
    template<> struct H5PredTypeSelector<unsigned int> { static H5::PredType type() { return H5::PredType::NATIVE_UINT; } };
    template<> struct H5PredTypeSelector<unsigned short> { static H5::PredType type() { return H5::PredType::NATIVE_USHORT; } };
    template<> struct H5PredTypeSelector<float> { static H5::PredType type() { return H5::PredType::NATIVE_FLOAT; } };
    template<> struct H5PredTypeSelector<double> { static H5::PredType type() { return H5::PredType::NATIVE_DOUBLE; } };
    template<> struct H5PredTypeSelector<bool> { 
        // HDF5无原生bool，使用unsigned char存储
        static H5::PredType type() { return H5::PredType::NATIVE_UCHAR; }
    };

    // 对std::size_t可能需要特殊处理，一般为unsigned long或unsigned long long
    // 假设为unsigned long long类型(64-bit)，可与NATIVE_ULLONG匹配
    template<> struct H5PredTypeSelector<std::size_t> {
        static H5::PredType type() { return H5::PredType::NATIVE_ULLONG; }
    };

    // 对字符串属性使用可变长字符串类型
    // 定义一个帮助函数，用于创建并写入字符串属性
    inline void writeStringAttribute(H5::H5Object& obj, const std::string& name, const std::string& value) {
        H5::DataSpace scalar_space(H5S_SCALAR);
        H5::StrType str_type(0, H5T_VARIABLE); // 可变长度字符串
        H5::Attribute attr = obj.createAttribute(name, str_type, scalar_space);
        attr.write(str_type, value);
    }

    // 写标量属性
    template <typename T>
    void defineAttributeScalar(H5::H5Object& obj, const std::string& name, const T& value) {
        auto ptype = H5PredTypeSelector<T>::type();
        H5::DataSpace scalar_space(H5S_SCALAR);
        H5::Attribute attr = obj.createAttribute(name, ptype, scalar_space);

        // 对bool进行特殊处理（bool存成unsigned char）
        if constexpr (std::is_same_v<T, bool>) {
            unsigned char val = value ? 1 : 0;
            attr.write(ptype, &val);
        } else {
            attr.write(ptype, &value);
        }
    }

    // 写向量属性
    template <typename T>
    void defineAttributeVector(H5::H5Object& obj, const std::string& name, const std::vector<T>& vec) {
        auto ptype = H5PredTypeSelector<T>::type();
        hsize_t dims[1] = { vec.size() };
        H5::DataSpace space(1, dims);
        H5::Attribute attr = obj.createAttribute(name, ptype, space);

        // 对bool向量特殊处理
        if constexpr (std::is_same_v<T, bool>) {
            std::vector<unsigned char> bool_data(vec.size());
            for (size_t i = 0; i < vec.size(); ++i) {
                bool_data[i] = vec[i] ? 1 : 0;
            }
            attr.write(ptype, bool_data.data());
        } else {
            attr.write(ptype, vec.data());
        }
    }

    // 对字符串向量属性，需要使用H5的可变长字符串数组
    inline void defineAttributeStringVector(H5::H5Object& obj, const std::string& name, const std::vector<std::string>& vec) {
        // 对于字符串数组属性，需要创建可变长字符串类型的数组。
        // 这比较复杂，需要创建H5::StrType，然后创建1D dataspace。
        hsize_t dims[1] = { vec.size() };
        H5::DataSpace space(1, dims);

        // 创建可变长字符串类型
        H5::StrType str_type(0, H5T_VARIABLE);
        H5::Attribute attr = obj.createAttribute(name, str_type, space);

        // HDF5的可变长字符串数组写入需要一个const char*数组
        std::vector<const char*> cstrs(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            cstrs[i] = vec[i].c_str();
        }

        attr.write(str_type, cstrs.data());
    }

} // namespace detail

// 针对标量类型的定义属性模板函数
template <typename T>
void defineAttribute(H5::H5Object& obj, const std::string& name, const std::any& value) {
    detail::defineAttributeScalar<T>(obj, name, std::any_cast<T>(value));
}

// 针对向量类型的定义属性模板函数
// 由于原代码写法不直接支持函数模板偏特化，这里使用重载
template <typename T>
void defineAttributeVectorType(H5::H5Object& obj, const std::string& name, const std::any& value) {
    auto v = std::any_cast<std::vector<T>>(value);
    detail::defineAttributeVector<T>(obj, name, v);
}

inline void defineAttributeStringVectorType(H5::H5Object& obj, const std::string& name, const std::any& value) {
    auto v = std::any_cast<std::vector<std::string>>(value);
    detail::defineAttributeStringVector(obj, name, v);
}

// 针对std::string标量的特化
template <>
void defineAttribute<std::string>(H5::H5Object& obj, const std::string& name, const std::any& value) {
    auto s = std::any_cast<std::string>(value);
    detail::writeStringAttribute(obj, name, s);
}

// 定义一个函数指针映射，用于writeAnyAttr
inline std::unordered_map<std::type_index, std::function<void(H5::H5Object&, const std::string&, const std::any&)>> buildHandlers() {
    using namespace std;
    unordered_map<type_index, function<void(H5::H5Object&, const string&, const any&)>> handlers;

    handlers[typeid(int)] = defineAttribute<int>;
    handlers[typeid(short)] = defineAttribute<short>;
    handlers[typeid(unsigned int)] = defineAttribute<unsigned int>;
    handlers[typeid(std::size_t)] = defineAttribute<std::size_t>;
    handlers[typeid(unsigned short)] = defineAttribute<unsigned short>;
    handlers[typeid(float)] = defineAttribute<float>;
    handlers[typeid(double)] = defineAttribute<double>;
    handlers[typeid(std::string)] = defineAttribute<std::string>;
    handlers[typeid(bool)] = defineAttribute<bool>;

    handlers[typeid(std::vector<int>)] = defineAttributeVectorType<int>;
    handlers[typeid(std::vector<short>)] = defineAttributeVectorType<short>;
    handlers[typeid(std::vector<unsigned int>)] = defineAttributeVectorType<unsigned int>;
    handlers[typeid(std::vector<std::size_t>)] = defineAttributeVectorType<std::size_t>;
    handlers[typeid(std::vector<unsigned short>)] = defineAttributeVectorType<unsigned short>;
    handlers[typeid(std::vector<float>)] = defineAttributeVectorType<float>;
    handlers[typeid(std::vector<double>)] = defineAttributeVectorType<double>;
    handlers[typeid(std::vector<std::string>)] = defineAttributeStringVectorType;
    handlers[typeid(std::vector<bool>)] = defineAttributeVectorType<bool>;

    return handlers;
}

inline void writeAnyAttr(H5::H5Object& obj, const std::string& name, const std::any& value) {
    static auto handlers = buildHandlers();

    auto it = handlers.find(value.type());
    if (it != handlers.end()) {
        it->second(obj, name, value);
    } else {
        throw std::runtime_error("Unsupported type in writeAnyAttr");
    }
}

} // namespace out

#endif // OUTPUT_UTILS_ATTR_WRITER_H
