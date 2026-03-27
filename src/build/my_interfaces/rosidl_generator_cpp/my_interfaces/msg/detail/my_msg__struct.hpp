// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from my_interfaces:msg/MyMsg.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "my_interfaces/msg/my_msg.hpp"


#ifndef MY_INTERFACES__MSG__DETAIL__MY_MSG__STRUCT_HPP_
#define MY_INTERFACES__MSG__DETAIL__MY_MSG__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__my_interfaces__msg__MyMsg __attribute__((deprecated))
#else
# define DEPRECATED__my_interfaces__msg__MyMsg __declspec(deprecated)
#endif

namespace my_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MyMsg_
{
  using Type = MyMsg_<ContainerAllocator>;

  explicit MyMsg_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->name = "";
      this->email = "";
      this->mssv = 0ll;
    }
  }

  explicit MyMsg_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : name(_alloc),
    email(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->name = "";
      this->email = "";
      this->mssv = 0ll;
    }
  }

  // field types and members
  using _name_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _name_type name;
  using _email_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _email_type email;
  using _mssv_type =
    int64_t;
  _mssv_type mssv;

  // setters for named parameter idiom
  Type & set__name(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->name = _arg;
    return *this;
  }
  Type & set__email(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->email = _arg;
    return *this;
  }
  Type & set__mssv(
    const int64_t & _arg)
  {
    this->mssv = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    my_interfaces::msg::MyMsg_<ContainerAllocator> *;
  using ConstRawPtr =
    const my_interfaces::msg::MyMsg_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<my_interfaces::msg::MyMsg_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<my_interfaces::msg::MyMsg_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::MyMsg_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::MyMsg_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      my_interfaces::msg::MyMsg_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<my_interfaces::msg::MyMsg_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<my_interfaces::msg::MyMsg_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<my_interfaces::msg::MyMsg_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__my_interfaces__msg__MyMsg
    std::shared_ptr<my_interfaces::msg::MyMsg_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__my_interfaces__msg__MyMsg
    std::shared_ptr<my_interfaces::msg::MyMsg_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MyMsg_ & other) const
  {
    if (this->name != other.name) {
      return false;
    }
    if (this->email != other.email) {
      return false;
    }
    if (this->mssv != other.mssv) {
      return false;
    }
    return true;
  }
  bool operator!=(const MyMsg_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MyMsg_

// alias to use template instance with default allocator
using MyMsg =
  my_interfaces::msg::MyMsg_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace my_interfaces

#endif  // MY_INTERFACES__MSG__DETAIL__MY_MSG__STRUCT_HPP_
