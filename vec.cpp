#include "vec.h"

float vec3::length() const {
    return sqrt(pow(x, 2.f) + pow(y, 2.f) + pow(z, 2.f));
}

float vec3::dot(const vec3& rhs) const {
    return x * rhs.x + y * rhs.y + z * rhs.z;
}