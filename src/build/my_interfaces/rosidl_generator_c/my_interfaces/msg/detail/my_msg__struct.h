// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from my_interfaces:msg/MyMsg.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "my_interfaces/msg/my_msg.h"


#ifndef MY_INTERFACES__MSG__DETAIL__MY_MSG__STRUCT_H_
#define MY_INTERFACES__MSG__DETAIL__MY_MSG__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

// Include directives for member types
// Member 'name'
// Member 'email'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/MyMsg in the package my_interfaces.
typedef struct my_interfaces__msg__MyMsg
{
  rosidl_runtime_c__String name;
  rosidl_runtime_c__String email;
  int64_t mssv;
} my_interfaces__msg__MyMsg;

// Struct for a sequence of my_interfaces__msg__MyMsg.
typedef struct my_interfaces__msg__MyMsg__Sequence
{
  my_interfaces__msg__MyMsg * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} my_interfaces__msg__MyMsg__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // MY_INTERFACES__MSG__DETAIL__MY_MSG__STRUCT_H_
