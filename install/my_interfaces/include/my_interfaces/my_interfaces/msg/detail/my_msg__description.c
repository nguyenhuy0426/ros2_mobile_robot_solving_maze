// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from my_interfaces:msg/MyMsg.idl
// generated code does not contain a copyright notice

#include "my_interfaces/msg/detail/my_msg__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_my_interfaces
const rosidl_type_hash_t *
my_interfaces__msg__MyMsg__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x41, 0xd7, 0x5e, 0xbc, 0xdb, 0x13, 0x37, 0xdc,
      0x83, 0xc4, 0xec, 0x29, 0xc9, 0xc4, 0x0c, 0xaa,
      0x47, 0x43, 0x60, 0xbe, 0x82, 0xc3, 0x2c, 0x8f,
      0x51, 0xb1, 0xf2, 0x1a, 0x40, 0xa3, 0xbb, 0xd7,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types

// Hashes for external referenced types
#ifndef NDEBUG
#endif

static char my_interfaces__msg__MyMsg__TYPE_NAME[] = "my_interfaces/msg/MyMsg";

// Define type names, field names, and default values
static char my_interfaces__msg__MyMsg__FIELD_NAME__name[] = "name";
static char my_interfaces__msg__MyMsg__FIELD_NAME__email[] = "email";
static char my_interfaces__msg__MyMsg__FIELD_NAME__mssv[] = "mssv";

static rosidl_runtime_c__type_description__Field my_interfaces__msg__MyMsg__FIELDS[] = {
  {
    {my_interfaces__msg__MyMsg__FIELD_NAME__name, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {my_interfaces__msg__MyMsg__FIELD_NAME__email, 5, 5},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
  {
    {my_interfaces__msg__MyMsg__FIELD_NAME__mssv, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT64,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
my_interfaces__msg__MyMsg__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {my_interfaces__msg__MyMsg__TYPE_NAME, 23, 23},
      {my_interfaces__msg__MyMsg__FIELDS, 3, 3},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "string name\n"
  "string email\n"
  "int64 mssv";

static char msg_encoding[] = "msg";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
my_interfaces__msg__MyMsg__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {my_interfaces__msg__MyMsg__TYPE_NAME, 23, 23},
    {msg_encoding, 3, 3},
    {toplevel_type_raw_source, 35, 35},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
my_interfaces__msg__MyMsg__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *my_interfaces__msg__MyMsg__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}
