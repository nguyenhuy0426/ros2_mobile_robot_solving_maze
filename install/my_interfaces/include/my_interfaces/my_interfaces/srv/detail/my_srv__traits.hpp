// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from my_interfaces:srv/MySrv.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "my_interfaces/srv/my_srv.hpp"


#ifndef MY_INTERFACES__SRV__DETAIL__MY_SRV__TRAITS_HPP_
#define MY_INTERFACES__SRV__DETAIL__MY_SRV__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "my_interfaces/srv/detail/my_srv__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace my_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const MySrv_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: vx
  {
    out << "vx: ";
    rosidl_generator_traits::value_to_yaml(msg.vx, out);
    out << ", ";
  }

  // member: vy
  {
    out << "vy: ";
    rosidl_generator_traits::value_to_yaml(msg.vy, out);
    out << ", ";
  }

  // member: wz
  {
    out << "wz: ";
    rosidl_generator_traits::value_to_yaml(msg.wz, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MySrv_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: vx
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vx: ";
    rosidl_generator_traits::value_to_yaml(msg.vx, out);
    out << "\n";
  }

  // member: vy
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "vy: ";
    rosidl_generator_traits::value_to_yaml(msg.vy, out);
    out << "\n";
  }

  // member: wz
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "wz: ";
    rosidl_generator_traits::value_to_yaml(msg.wz, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MySrv_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace my_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use my_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const my_interfaces::srv::MySrv_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  my_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use my_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const my_interfaces::srv::MySrv_Request & msg)
{
  return my_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<my_interfaces::srv::MySrv_Request>()
{
  return "my_interfaces::srv::MySrv_Request";
}

template<>
inline const char * name<my_interfaces::srv::MySrv_Request>()
{
  return "my_interfaces/srv/MySrv_Request";
}

template<>
struct has_fixed_size<my_interfaces::srv::MySrv_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<my_interfaces::srv::MySrv_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<my_interfaces::srv::MySrv_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace my_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const MySrv_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: phi
  {
    if (msg.phi.size() == 0) {
      out << "phi: []";
    } else {
      out << "phi: [";
      size_t pending_items = msg.phi.size();
      for (auto item : msg.phi) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: ok
  {
    out << "ok: ";
    rosidl_generator_traits::value_to_yaml(msg.ok, out);
    out << ", ";
  }

  // member: message
  {
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MySrv_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: phi
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.phi.size() == 0) {
      out << "phi: []\n";
    } else {
      out << "phi:\n";
      for (auto item : msg.phi) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: ok
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ok: ";
    rosidl_generator_traits::value_to_yaml(msg.ok, out);
    out << "\n";
  }

  // member: message
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MySrv_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace my_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use my_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const my_interfaces::srv::MySrv_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  my_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use my_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const my_interfaces::srv::MySrv_Response & msg)
{
  return my_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<my_interfaces::srv::MySrv_Response>()
{
  return "my_interfaces::srv::MySrv_Response";
}

template<>
inline const char * name<my_interfaces::srv::MySrv_Response>()
{
  return "my_interfaces/srv/MySrv_Response";
}

template<>
struct has_fixed_size<my_interfaces::srv::MySrv_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<my_interfaces::srv::MySrv_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<my_interfaces::srv::MySrv_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__traits.hpp"

namespace my_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const MySrv_Event & msg,
  std::ostream & out)
{
  out << "{";
  // member: info
  {
    out << "info: ";
    to_flow_style_yaml(msg.info, out);
    out << ", ";
  }

  // member: request
  {
    if (msg.request.size() == 0) {
      out << "request: []";
    } else {
      out << "request: [";
      size_t pending_items = msg.request.size();
      for (auto item : msg.request) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: response
  {
    if (msg.response.size() == 0) {
      out << "response: []";
    } else {
      out << "response: [";
      size_t pending_items = msg.response.size();
      for (auto item : msg.response) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const MySrv_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: info
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "info:\n";
    to_block_style_yaml(msg.info, out, indentation + 2);
  }

  // member: request
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.request.size() == 0) {
      out << "request: []\n";
    } else {
      out << "request:\n";
      for (auto item : msg.request) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: response
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.response.size() == 0) {
      out << "response: []\n";
    } else {
      out << "response:\n";
      for (auto item : msg.response) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const MySrv_Event & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace my_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use my_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const my_interfaces::srv::MySrv_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  my_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use my_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const my_interfaces::srv::MySrv_Event & msg)
{
  return my_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<my_interfaces::srv::MySrv_Event>()
{
  return "my_interfaces::srv::MySrv_Event";
}

template<>
inline const char * name<my_interfaces::srv::MySrv_Event>()
{
  return "my_interfaces/srv/MySrv_Event";
}

template<>
struct has_fixed_size<my_interfaces::srv::MySrv_Event>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<my_interfaces::srv::MySrv_Event>
  : std::integral_constant<bool, has_bounded_size<my_interfaces::srv::MySrv_Request>::value && has_bounded_size<my_interfaces::srv::MySrv_Response>::value && has_bounded_size<service_msgs::msg::ServiceEventInfo>::value> {};

template<>
struct is_message<my_interfaces::srv::MySrv_Event>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<my_interfaces::srv::MySrv>()
{
  return "my_interfaces::srv::MySrv";
}

template<>
inline const char * name<my_interfaces::srv::MySrv>()
{
  return "my_interfaces/srv/MySrv";
}

template<>
struct has_fixed_size<my_interfaces::srv::MySrv>
  : std::integral_constant<
    bool,
    has_fixed_size<my_interfaces::srv::MySrv_Request>::value &&
    has_fixed_size<my_interfaces::srv::MySrv_Response>::value
  >
{
};

template<>
struct has_bounded_size<my_interfaces::srv::MySrv>
  : std::integral_constant<
    bool,
    has_bounded_size<my_interfaces::srv::MySrv_Request>::value &&
    has_bounded_size<my_interfaces::srv::MySrv_Response>::value
  >
{
};

template<>
struct is_service<my_interfaces::srv::MySrv>
  : std::true_type
{
};

template<>
struct is_service_request<my_interfaces::srv::MySrv_Request>
  : std::true_type
{
};

template<>
struct is_service_response<my_interfaces::srv::MySrv_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // MY_INTERFACES__SRV__DETAIL__MY_SRV__TRAITS_HPP_
