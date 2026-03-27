// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from my_interfaces:msg/MyMsg.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "my_interfaces/msg/my_msg.hpp"


#ifndef MY_INTERFACES__MSG__DETAIL__MY_MSG__TRAITS_HPP_
#define MY_INTERFACES__MSG__DETAIL__MY_MSG__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "my_interfaces/msg/detail/my_msg__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace my_interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const MyMsg & msg,
  std::ostream & out)
{
  out << "{";
  // member: name
  {
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << ", ";
  }

  // member: email
  {
    out << "email: ";
    rosidl_generator_traits::value_to_yaml(msg.email, out);
    out << ", ";
  }

  // member: mssv
  {
    out << "mssv: ";
    rosidl_generator_traits::value_to_yaml(msg.mssv, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MyMsg & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << "\n";
  }

  // member: email
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "email: ";
    rosidl_generator_traits::value_to_yaml(msg.email, out);
    out << "\n";
  }

  // member: mssv
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mssv: ";
    rosidl_generator_traits::value_to_yaml(msg.mssv, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MyMsg & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace my_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use my_interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const my_interfaces::msg::MyMsg & msg,
  std::ostream & out, size_t indentation = 0)
{
  my_interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use my_interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const my_interfaces::msg::MyMsg & msg)
{
  return my_interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<my_interfaces::msg::MyMsg>()
{
  return "my_interfaces::msg::MyMsg";
}

template<>
inline const char * name<my_interfaces::msg::MyMsg>()
{
  return "my_interfaces/msg/MyMsg";
}

template<>
struct has_fixed_size<my_interfaces::msg::MyMsg>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<my_interfaces::msg::MyMsg>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<my_interfaces::msg::MyMsg>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // MY_INTERFACES__MSG__DETAIL__MY_MSG__TRAITS_HPP_
