#ifndef BEM_GPU_SELECT_H
#define BEM_GPU_SELECT_H

#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

inline bool bem_ascii_token_equal(const char* begin, const char* end, const char* token)
{
    size_t len = (size_t)(end - begin);
    if (len != std::strlen(token))
        return false;
    for (size_t i = 0; i < len; ++i) {
        unsigned char a = (unsigned char)begin[i];
        unsigned char b = (unsigned char)token[i];
        if (std::tolower(a) != std::tolower(b))
            return false;
    }
    return true;
}

inline bool bem_env_value_enabled(const char* value, bool default_value = false)
{
    if (!value)
        return default_value;
    const char* begin = value;
    while (*begin && std::isspace((unsigned char)*begin))
        ++begin;
    const char* end = begin + std::strlen(begin);
    while (end > begin && std::isspace((unsigned char)*(end - 1)))
        --end;
    if (end == begin)
        return true;
    if (bem_ascii_token_equal(begin, end, "0") ||
        bem_ascii_token_equal(begin, end, "false") ||
        bem_ascii_token_equal(begin, end, "no") ||
        bem_ascii_token_equal(begin, end, "off")) {
        return false;
    }
    if (bem_ascii_token_equal(begin, end, "1") ||
        bem_ascii_token_equal(begin, end, "true") ||
        bem_ascii_token_equal(begin, end, "yes") ||
        bem_ascii_token_equal(begin, end, "on")) {
        return true;
    }
    return true;
}

inline bool bem_env_flag_enabled(const char* name, bool default_value = false)
{
    return bem_env_value_enabled(std::getenv(name), default_value);
}

inline bool bem_env_flag_present(const char* name)
{
    return std::getenv(name) != nullptr;
}

inline bool bem_env_has_value(const char* name)
{
    const char* value = std::getenv(name);
    if (!value)
        return false;
    while (*value) {
        if (!std::isspace((unsigned char)*value))
            return true;
        ++value;
    }
    return false;
}

inline bool bem_parse_int_value(const char* value, int* out)
{
    if (!value || !out)
        return false;
    while (*value && std::isspace((unsigned char)*value))
        ++value;
    if (!*value)
        return false;
    errno = 0;
    char* end = 0;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || errno == ERANGE || parsed < INT_MIN || parsed > INT_MAX)
        return false;
    while (*end && std::isspace((unsigned char)*end))
        ++end;
    if (*end)
        return false;
    *out = (int)parsed;
    return true;
}

inline int bem_env_int(const char* name, int default_value)
{
    int parsed = default_value;
    if (bem_parse_int_value(std::getenv(name), &parsed))
        return parsed;
    return default_value;
}

inline bool bem_parse_double_value(const char* value, double* out)
{
    if (!value || !out)
        return false;
    while (*value && std::isspace((unsigned char)*value))
        ++value;
    if (!*value)
        return false;
    errno = 0;
    char* end = 0;
    double parsed = std::strtod(value, &end);
    if (end == value || errno == ERANGE || !std::isfinite(parsed))
        return false;
    while (*end && std::isspace((unsigned char)*end))
        ++end;
    if (*end)
        return false;
    *out = parsed;
    return true;
}

inline double bem_env_double(const char* name, double default_value)
{
    double parsed = default_value;
    if (bem_parse_double_value(std::getenv(name), &parsed))
        return parsed;
    return default_value;
}

inline std::vector<int> bem_parse_gpu_list_env(const char* text)
{
    std::vector<int> devices;
    if (!text || !*text)
        return devices;

    const char* p = text;
    while (*p) {
        while (*p && (std::isspace((unsigned char)*p) || *p == ','))
            ++p;
        if (!*p)
            break;
        char* end = 0;
        long value = std::strtol(p, &end, 10);
        if (end == p)
            return std::vector<int>();
        if (value < 0)
            return std::vector<int>();
        devices.push_back((int)value);
        p = end;
    }
    return devices;
}

inline bool bem_validate_gpu_list(const std::vector<int>& devices, int device_count)
{
    if (devices.empty() || device_count <= 0)
        return false;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i] < 0 || devices[i] >= device_count)
            return false;
        for (size_t j = i + 1; j < devices.size(); ++j) {
            if (devices[i] == devices[j])
                return false;
        }
    }
    return true;
}

#endif // BEM_GPU_SELECT_H
