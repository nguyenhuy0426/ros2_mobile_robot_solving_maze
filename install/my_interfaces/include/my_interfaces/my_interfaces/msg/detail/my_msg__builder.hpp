// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from my_interfaces:msg/MyMsg.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "my_interfaces/msg/my_msg.hpp"


#ifndef MY_INTERFACES__MSG__DETAIL__MY_MSG__BUILDER_HPP_
#define MY_INTERFACES__MSG__DETAIL__MY_MSG__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "my_interfaces/msg/detail/my_msg__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace my_interfaces
{

namespace msg
{

namespace builder
{

class Init_MyMsg_mssv
{
public:
  explicit Init_MyMsg_mssv(::my_interfaces::msg::MyMsg & msg)
  : msg_(msg)
  {}
  ::my_interfaces::msg::MyMsg mssv(::my_interfaces::msg::MyMsg::_mssv_type arg)
  {
    msg_.mssv = std::move(arg);
    return std::move(msg_);
  }

private:
  ::my_interfaces::msg::MyMsg msg_;
};

class Init_MyMsg_email
{
public:
  explicit Init_MyMsg_email(::my_interfaces::msg::MyMsg & msg)
  : msg_(msg)
  {}
  Init_MyMsg_mssv email(::my_interfaces::msg::MyMsg::_email_type arg)
  {
    msg_.email = std::move(arg);
    return Init_MyMsg_mssv(msg_);
  }

private:
  ::my_interfaces::msg::MyMsg msg_;
};

class Init_MyMsg_name
{
public:
  Init_MyMsg_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MyMsg_email name(::my_interfaces::msg::MyMsg::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_MyMsg_email(msg_);
  }

private:
  ::my_interfaces::msg::MyMsg msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::my_interfaces::msg::MyMsg>()
{
  return my_interfaces::msg::builder::Init_MyMsg_name();
}

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__MY_MSG__BUILDER_HPP_
