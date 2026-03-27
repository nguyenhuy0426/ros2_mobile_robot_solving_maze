// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from my_interfaces:srv/MySrv.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "my_interfaces/srv/my_srv.hpp"


#ifndef MY_INTERFACES__SRV__DETAIL__MY_SRV__BUILDER_HPP_
#define MY_INTERFACES__SRV__DETAIL__MY_SRV__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "my_interfaces/srv/detail/my_srv__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace my_interfaces
{

namespace srv
{

namespace builder
{

class Init_MySrv_Request_wz
{
public:
  explicit Init_MySrv_Request_wz(::my_interfaces::srv::MySrv_Request & msg)
  : msg_(msg)
  {}
  ::my_interfaces::srv::MySrv_Request wz(::my_interfaces::srv::MySrv_Request::_wz_type arg)
  {
    msg_.wz = std::move(arg);
    return std::move(msg_);
  }

private:
  ::my_interfaces::srv::MySrv_Request msg_;
};

class Init_MySrv_Request_vy
{
public:
  explicit Init_MySrv_Request_vy(::my_interfaces::srv::MySrv_Request & msg)
  : msg_(msg)
  {}
  Init_MySrv_Request_wz vy(::my_interfaces::srv::MySrv_Request::_vy_type arg)
  {
    msg_.vy = std::move(arg);
    return Init_MySrv_Request_wz(msg_);
  }

private:
  ::my_interfaces::srv::MySrv_Request msg_;
};

class Init_MySrv_Request_vx
{
public:
  Init_MySrv_Request_vx()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MySrv_Request_vy vx(::my_interfaces::srv::MySrv_Request::_vx_type arg)
  {
    msg_.vx = std::move(arg);
    return Init_MySrv_Request_vy(msg_);
  }

private:
  ::my_interfaces::srv::MySrv_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::my_interfaces::srv::MySrv_Request>()
{
  return my_interfaces::srv::builder::Init_MySrv_Request_vx();
}

}  // namespace my_interfaces


namespace my_interfaces
{

namespace srv
{

namespace builder
{

class Init_MySrv_Response_message
{
public:
  explicit Init_MySrv_Response_message(::my_interfaces::srv::MySrv_Response & msg)
  : msg_(msg)
  {}
  ::my_interfaces::srv::MySrv_Response message(::my_interfaces::srv::MySrv_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::my_interfaces::srv::MySrv_Response msg_;
};

class Init_MySrv_Response_ok
{
public:
  explicit Init_MySrv_Response_ok(::my_interfaces::srv::MySrv_Response & msg)
  : msg_(msg)
  {}
  Init_MySrv_Response_message ok(::my_interfaces::srv::MySrv_Response::_ok_type arg)
  {
    msg_.ok = std::move(arg);
    return Init_MySrv_Response_message(msg_);
  }

private:
  ::my_interfaces::srv::MySrv_Response msg_;
};

class Init_MySrv_Response_phi
{
public:
  Init_MySrv_Response_phi()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MySrv_Response_ok phi(::my_interfaces::srv::MySrv_Response::_phi_type arg)
  {
    msg_.phi = std::move(arg);
    return Init_MySrv_Response_ok(msg_);
  }

private:
  ::my_interfaces::srv::MySrv_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::my_interfaces::srv::MySrv_Response>()
{
  return my_interfaces::srv::builder::Init_MySrv_Response_phi();
}

}  // namespace my_interfaces


namespace my_interfaces
{

namespace srv
{

namespace builder
{

class Init_MySrv_Event_response
{
public:
  explicit Init_MySrv_Event_response(::my_interfaces::srv::MySrv_Event & msg)
  : msg_(msg)
  {}
  ::my_interfaces::srv::MySrv_Event response(::my_interfaces::srv::MySrv_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::my_interfaces::srv::MySrv_Event msg_;
};

class Init_MySrv_Event_request
{
public:
  explicit Init_MySrv_Event_request(::my_interfaces::srv::MySrv_Event & msg)
  : msg_(msg)
  {}
  Init_MySrv_Event_response request(::my_interfaces::srv::MySrv_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_MySrv_Event_response(msg_);
  }

private:
  ::my_interfaces::srv::MySrv_Event msg_;
};

class Init_MySrv_Event_info
{
public:
  Init_MySrv_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MySrv_Event_request info(::my_interfaces::srv::MySrv_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_MySrv_Event_request(msg_);
  }

private:
  ::my_interfaces::srv::MySrv_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::my_interfaces::srv::MySrv_Event>()
{
  return my_interfaces::srv::builder::Init_MySrv_Event_info();
}

}  // namespace my_interfaces

#endif  // MY_INTERFACES__SRV__DETAIL__MY_SRV__BUILDER_HPP_
