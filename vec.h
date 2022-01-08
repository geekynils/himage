#pragma once

#ifndef VEC_H
#define VEC_H

#include <cmath>

struct vec3 {

    float x;
    float y;
    float z;

    float length() const;

    float dot(const vec3& rhs) const;
};

#endif